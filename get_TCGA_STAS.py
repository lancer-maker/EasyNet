import os
import re
import time
import hashlib
import requests
import pandas as pd
from tqdm import tqdm

# ============================================================
# 1. 下载保存路径
# ============================================================
DOWNLOAD_DIR = r"D:\\files\\HE\\TCGA_STAS\\TCGA_STAD_SVS_GeoJSON"
LOG_FILE = os.path.join(os.path.dirname(DOWNLOAD_DIR), "download.log")
MANIFEST_FILE = os.path.join(DOWNLOAD_DIR, "tcga_stad_svs_manifest.csv")

# GDC API
GDC_FILES_ENDPT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPT = "https://api.gdc.cancer.gov/data/"
HEADERS = {"User-Agent": "TCGA-STAD-SVS-Downloader/2.0"}

# 下载参数
MAX_RETRIES = 5
RETRY_DELAY = 15
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB
PAGE_SIZE = 1000
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 600

# GDC 项目页当前显示 Diagnostic Slide 442 个 + Tissue Slide 755 个。
PORTAL_REFERENCE_COUNT = 1197
TARGET_STRATEGIES = ["Diagnostic Slide", "Tissue Slide"]


class DownloadIntegrityError(Exception):
    """下载文件大小、Range 或校验和不符合预期。"""


def log(message):
    """同时写入日志文件并实时显示在终端；兼容 tqdm 进度条。"""
    message = str(message)
    try:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except OSError:
        pass
    tqdm.write(message)


def request_with_status(method, url, **kwargs):
    """发送 HTTP 请求，并实时输出请求地址、最终地址、状态码或连接错误。"""
    log(f"[连接中] {method.upper()} {url}")
    try:
        response = requests.request(method=method, url=url, **kwargs)
        log(
            f"[HTTP响应] {method.upper()} {response.url} | "
            f"状态码: {response.status_code} {response.reason}"
        )
        return response
    except requests.RequestException as exc:
        log(f"[连接失败] {method.upper()} {url} | {type(exc).__name__}: {exc}")
        raise


def normalize_size(value):
    """把文件大小转为正整数；缺失或无效时返回 None。"""
    try:
        value = int(value)
        return value if value > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def extract_project_ids(hit):
    """兼容 GDC 文件结果中 cases/project 的常见嵌套结构。"""
    project_ids = []
    cases = hit.get("cases", [])
    if isinstance(cases, dict):
        cases = [cases]
    if not isinstance(cases, list):
        return project_ids

    for case in cases:
        if not isinstance(case, dict):
            continue
        project = case.get("project", {})
        if isinstance(project, dict):
            project_id = project.get("project_id")
            if project_id:
                project_ids.append(str(project_id))
    return project_ids


def get_svs_files():
    """分页查询 TCGA-STAD 项目下的 Diagnostic/Tissue Slide SVS 文件。"""
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": "TCGA-STAD",
                },
            },
            {
                "op": "=",
                "content": {"field": "data_format", "value": "SVS"},
            },
            {
                "op": "in",
                "content": {
                    "field": "experimental_strategy",
                    "value": TARGET_STRATEGIES,
                },
            },
        ],
    }

    fields = (
        "file_id,file_name,file_size,md5sum,data_format,"
        "experimental_strategy,data_type,cases.project.project_id"
    )

    log("开始查询 GDC：项目 TCGA-STAD、格式 SVS、切片类型 Diagnostic/Tissue Slide。")

    all_hits = []
    offset = 0
    api_total = None

    while api_total is None or offset < api_total:
        payload = {
            "filters": filters,
            "format": "JSON",
            "fields": fields,
            "size": PAGE_SIZE,
            "from": offset,
        }

        with request_with_status(
            "POST",
            GDC_FILES_ENDPT,
            json=payload,
            headers=HEADERS,
            timeout=(CONNECT_TIMEOUT, 120),
        ) as response:
            response.raise_for_status()
            body = response.json()

        data = body.get("data", {})
        hits = data.get("hits", [])
        pagination = data.get("pagination", {})
        total_value = pagination.get("total")
        if total_value is not None:
            api_total = int(total_value)

        log(
            f"[清单分页] 本页 {len(hits)} 个；"
            f"已读取 {offset + len(hits)} 个；"
            f"API 总数 {api_total if api_total is not None else '未知'}"
        )

        if not hits:
            break

        all_hits.extend(hits)
        offset += len(hits)

        if api_total is None and len(hits) < PAGE_SIZE:
            break

    if not all_hits:
        raise RuntimeError(
            "没有查询到符合条件的 SVS 文件。请检查 GDC API 返回内容或筛选字段。"
        )

    df = pd.DataFrame(all_hits)

    if "file_id" not in df.columns:
        df["file_id"] = df["id"] if "id" in df.columns else None
    elif "id" in df.columns:
        df["file_id"] = df["file_id"].fillna(df["id"])

    if "file_size" not in df.columns and "size" in df.columns:
        df = df.rename(columns={"size": "file_size"})
    for column in [
        "file_name", "file_size", "md5sum", "data_format",
        "experimental_strategy", "data_type"
    ]:
        if column not in df.columns:
            df[column] = None

    df["file_size"] = pd.to_numeric(df["file_size"], errors="coerce")
    df["file_id"] = df["file_id"].fillna("").astype(str)
    df["file_name"] = df["file_name"].fillna("").astype(str)
    df["md5sum"] = df["md5sum"].fillna("").astype(str).str.lower()

    df = df[(df["file_id"] != "") & (df["file_name"] != "")].copy()
    df = df.drop_duplicates(subset=["file_id"], keep="first")
    df = df.sort_values("file_size", na_position="last").reset_index(drop=True)

    observed_project_ids = set()
    for hit in all_hits:
        observed_project_ids.update(extract_project_ids(hit))
    unexpected = observed_project_ids - {"TCGA-STAD"}
    if unexpected:
        raise RuntimeError(
            f"API 结果中出现非 TCGA-STAD 项目：{sorted(unexpected)}。已停止以避免下载错误数据。"
        )

    wrong_format = df[
        df["data_format"].notna()
        & (df["data_format"].astype(str).str.upper() != "SVS")
    ]
    wrong_strategy = df[
        df["experimental_strategy"].notna()
        & ~df["experimental_strategy"].astype(str).isin(TARGET_STRATEGIES)
    ]
    if not wrong_format.empty or not wrong_strategy.empty:
        raise RuntimeError(
            "查询结果含非 SVS 或非 Diagnostic/Tissue Slide 条目，"
            "为避免误下载已终止。请查看日志。"
        )

    count = len(df)
    log(f"[查询完成] 去重后共 {count} 个 SVS 文件。")
    if count != PORTAL_REFERENCE_COUNT:
        log(
            f"[数量提示] 当前清单为 {count} 个；网页当前可见参考数为 "
            f"{PORTAL_REFERENCE_COUNT} 个（442 Diagnostic Slide + 755 Tissue Slide）。"
            "项目数据可能更新；请核对 API 筛选/分页结果，不会因数量不同而强制中止。"
        )
    else:
        log(f"[数量校验] 与网页参考总数 {PORTAL_REFERENCE_COUNT} 一致。")

    return df


def calculate_md5(file_path):
    """流式计算 MD5，不显示单独进度条。"""
    digest = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_completed_file(file_path, expected_size, expected_md5):
    """验证已存在或刚下载完成的文件，返回 (是否通过, 校验结果)。"""
    actual_size = os.path.getsize(file_path)

    if expected_size is not None and actual_size != expected_size:
        return False, f"size_mismatch:{actual_size}/{expected_size}"

    if expected_md5:
        actual_md5 = calculate_md5(file_path)
        if actual_md5.lower() != expected_md5.lower():
            return False, f"md5_mismatch:actual={actual_md5}, expected={expected_md5}"
        return True, "md5_verified"

    return True, "size_verified_no_md5"


def resume_download(file_id, file_name, expected_size, expected_md5, save_dir, progress=None):
    """下载单个文件；进度通过外层 progress 条统一显示。

    progress 仅在 expected_size 已知时上报字节，避免未知大小文件把进度条推超。
    重试或服务器忽略 Range 时会回退已经上报的字节，保证总体进度不超过 total。
    """
    os.makedirs(save_dir, exist_ok=True)
    safe_name = os.path.basename(str(file_name))
    file_path = os.path.join(save_dir, safe_name)
    expected_size = normalize_size(expected_size)
    expected_md5 = str(expected_md5 or "").strip().lower()
    url = GDC_DATA_ENDPT + str(file_id)

    tracked = {"reported": 0}

    def report(n):
        if progress is None or expected_size is None or n <= 0:
            return
        progress.update(n)
        tracked["reported"] += n

    def reset_report():
        if progress is not None and tracked["reported"]:
            progress.update(-tracked["reported"])
        tracked["reported"] = 0

    # 如果本地文件大小恰好正确，先校验 MD5，再决定是否跳过。
    if os.path.isfile(file_path) and expected_size is not None:
        local_size = os.path.getsize(file_path)
        if local_size == expected_size:
            passed, status = verify_completed_file(
                file_path, expected_size, expected_md5
            )
            if passed:
                log(f"[跳过] {safe_name} | 本地文件有效 | {status}")
                report(expected_size)
                return True, f"already_present_{status}"
            log(f"[本地文件无效] {safe_name} | {status} | 将重新下载。")
            os.remove(file_path)
        elif local_size > expected_size:
            log(
                f"[本地文件异常] {safe_name} | 本地 {local_size:,} B > "
                f"清单 {expected_size:,} B；删除后重新下载。"
            )
            os.remove(file_path)

    for attempt in range(1, MAX_RETRIES + 1):
        # 每次重试前回退上次尝试已上报的字节
        reset_report()

        if expected_size is not None and os.path.isfile(file_path):
            downloaded = os.path.getsize(file_path)
            if downloaded > expected_size:
                os.remove(file_path)
                downloaded = 0
        else:
            downloaded = 0

        # 续传时磁盘上已有的字节也计入总进度
        report(downloaded)

        request_headers = dict(HEADERS)
        if downloaded > 0:
            request_headers["Range"] = f"bytes={downloaded}-"
            log(
                f"[续传] {safe_name} | 从 {downloaded:,}/{expected_size:,} B 继续 | "
                f"尝试 {attempt}/{MAX_RETRIES}"
            )
        else:
            log(
                f"[开始下载] {safe_name} | "
                f"大小 {f'{expected_size:,} B' if expected_size else '未知'} | "
                f"尝试 {attempt}/{MAX_RETRIES}"
            )

        try:
            with request_with_status(
                "GET",
                url,
                headers=request_headers,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            ) as response:
                if response.status_code == 416:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise DownloadIntegrityError("HTTP 416 Range Not Satisfiable；已清理本地文件。")

                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range, re.I)
                    if not match or int(match.group(1)) != downloaded:
                        raise DownloadIntegrityError(
                            f"服务器 Content-Range 与请求偏移不符："
                            f"requested={downloaded}, received={content_range!r}"
                        )
                    mode = "ab" if downloaded > 0 else "wb"
                elif response.status_code == 200:
                    # 服务器不支持/忽略 Range：从头覆盖，之前上报的续传字节要回退
                    mode = "wb"
                    if downloaded > 0:
                        reset_report()
                    downloaded = 0
                else:
                    response.raise_for_status()
                    raise DownloadIntegrityError(
                        f"非预期 HTTP 状态码：{response.status_code}"
                    )

                log(
                    f"[下载连接正常] HTTP {response.status_code} | {safe_name} | "
                    f"Content-Length={response.headers.get('Content-Length', '未知')} | "
                    f"Content-Range={response.headers.get('Content-Range', '无')}"
                )

                with open(file_path, mode) as file_handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        file_handle.write(chunk)
                        report(len(chunk))
                    file_handle.flush()
                    os.fsync(file_handle.fileno())

            final_size = os.path.getsize(file_path)
            if expected_size is not None and final_size < expected_size:
                log(
                    f"[下载未完成] {safe_name} | {final_size:,}/{expected_size:,} B；"
                    "保留部分文件，下一次尝试继续续传。"
                )
                raise DownloadIntegrityError("响应结束但文件仍小于 GDC 清单大小。")

            if expected_size is not None and final_size > expected_size:
                log(
                    f"[下载大小异常] {safe_name} | {final_size:,} B > "
                    f"清单 {expected_size:,} B；删除后重试。"
                )
                os.remove(file_path)
                raise DownloadIntegrityError("下载文件大于 GDC 清单记录的大小。")

            passed, integrity_status = verify_completed_file(
                file_path, expected_size, expected_md5
            )
            if not passed:
                log(f"[校验失败] {safe_name} | {integrity_status}；删除后重试。")
                os.remove(file_path)
                raise DownloadIntegrityError(integrity_status)

            log(
                f"[下载完成] {safe_name} | {final_size:,} B | "
                f"完整性：{integrity_status}"
            )
            return True, integrity_status

        except (requests.RequestException, OSError, ValueError, DownloadIntegrityError) as exc:
            log(
                f"[下载错误] {safe_name} | 尝试 {attempt}/{MAX_RETRIES} | "
                f"{type(exc).__name__}: {exc}"
            )
        except Exception as exc:
            log(
                f"[未知错误] {safe_name} | 尝试 {attempt}/{MAX_RETRIES} | "
                f"{type(exc).__name__}: {exc}"
            )

        if attempt < MAX_RETRIES:
            delay = RETRY_DELAY * attempt
            log(f"[等待重试] {delay} 秒后重试：{safe_name}")
            time.sleep(delay)

    log(f"[最终失败] {safe_name} | 已达到最大重试次数。")
    reset_report()
    return False, "failed"


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    log("=" * 78)
    log("TCGA-STAD SVS 下载启动")
    log(f"保存目录：{DOWNLOAD_DIR}")
    log(f"日志文件：{LOG_FILE}")
    log(f"清单文件：{MANIFEST_FILE}")
    log("=" * 78)

    df = get_svs_files()
    df["download_status"] = "pending"
    df["integrity_status"] = "not_checked"
    df["local_path"] = df["file_name"].apply(
        lambda name: os.path.join(DOWNLOAD_DIR, os.path.basename(str(name)))
    )
    df.to_csv(MANIFEST_FILE, index=False, encoding="utf-8-sig")
    log(f"[清单已保存] {MANIFEST_FILE}")

    success_count = 0
    fail_count = 0
    verified_count = 0
    total = len(df)

    known_sizes = df["file_size"].dropna()
    known_sizes = known_sizes[known_sizes > 0]
    total_bytes = int(known_sizes.sum()) if not known_sizes.empty else 0
    use_byte_progress = total_bytes > 0

    with tqdm(
        total=total_bytes if use_byte_progress else total,
        desc="总下载进度",
        unit="B" if use_byte_progress else "file",
        unit_scale=use_byte_progress,
        unit_divisor=1024,
        position=0,
        dynamic_ncols=True,
        leave=True,
        mininterval=0.5,
    ) as overall:
        for i, row in df.iterrows():
            file_id = row["file_id"]
            file_name = row["file_name"]
            file_size = row["file_size"]
            md5sum = row["md5sum"]

            size_text = f"{int(file_size):,} B" if pd.notna(file_size) else "大小未知"
            log(f"\n[{i + 1}/{total}] {file_name} | {size_text} | File ID={file_id}")

            ok, integrity_status = resume_download(
                file_id=file_id,
                file_name=file_name,
                expected_size=file_size,
                expected_md5=md5sum,
                save_dir=DOWNLOAD_DIR,
                progress=overall if use_byte_progress else None,
            )

            if ok:
                success_count += 1
                df.at[i, "download_status"] = "success"
                df.at[i, "integrity_status"] = integrity_status
                if "md5_verified" in integrity_status:
                    verified_count += 1
            else:
                fail_count += 1
                df.at[i, "download_status"] = "failed"
                df.at[i, "integrity_status"] = integrity_status

            df.to_csv(MANIFEST_FILE, index=False, encoding="utf-8-sig")

            if not use_byte_progress:
                overall.update(1)
            overall.set_postfix(ok=success_count, failed=fail_count)

    log("\n" + "=" * 78)
    log("全部下载任务结束。")
    log(f"SVS 总数：{total}")
    log(f"成功：{success_count}")
    log(f"失败：{fail_count}")
    log(f"通过 GDC MD5 校验：{verified_count}")
    log(f"下载清单：{MANIFEST_FILE}")
    log(f"日志文件：{LOG_FILE}")
    log("注意：此脚本不把 GDC 的管理性 annotations 元数据误当成病理区域 GeoJSON。")
    log("=" * 78)


if __name__ == "__main__":
    main()