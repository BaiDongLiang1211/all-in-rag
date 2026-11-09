import os
import json
import logging
from collections import Counter
from pathlib import Path
from time import perf_counter

# 必须在导入 Hugging Face 相关依赖前设置模型下载镜像。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 项目内的工具与模型，避免依赖系统 PATH 或每次联网下载。
project_root = Path(__file__).resolve().parents[2]
tools_dir = project_root.parent / ".pdf-tools"
runtime_bin = tools_dir / "runtime/Library/bin"
if runtime_bin.is_dir():
    os.environ["PATH"] = str(runtime_bin) + os.pathsep + os.environ.get("PATH", "")
    os.environ["TESSDATA_PREFIX"] = str(tools_dir / "runtime/share/tessdata")

from unstructured.partition.pdf import partition_pdf

# 当前版本按空格匹配 OCR 文本与识别框，会漏掉不以空格分词的中文。
# 在本脚本进程内改用库提供的空间分组，保留中文识别框及其文本。
from unstructured.partition.pdf_image.inference_utils import build_layout_elements_from_ocr_regions
from unstructured.partition.utils.ocr_models.tesseract_ocr import OCRAgentTesseract


def group_ocr_by_position(self, image):
    return build_layout_elements_from_ocr_regions(
        ocr_regions=self.get_layout_from_image(image),
        group_by_ocr_text=False,
    )


OCRAgentTesseract.get_layout_elements_from_image = group_ocr_by_position

# 使用本地 YOLOX 权重；通过库提供的注册接口配置模型。
model_path = tools_dir / "models/yolox_l0.05.onnx"
if model_path.is_file():
    from unstructured_inference.models.base import register_new_model
    from unstructured_inference.models.yolox import UnstructuredYoloXModel, YOLOX_LABEL_MAP

    register_new_model(
        {"yolox": {"model_path": str(model_path), "label_map": YOLOX_LABEL_MAP}},
        UnstructuredYoloXModel,
    )

# 仅过滤已确认的字体描述警告，保留其他 PDF 解析警告。
class FontBBoxFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Could not get FontBBox from font descriptor")


logging.getLogger("pdfminer.pdffont").addFilter(FontBBoxFilter())

# PDF文件路径
pdf_path = project_root / "data/C2/pdf/rag.pdf"
output_dir = Path(__file__).resolve().parent / "partition_results"
output_dir.mkdir(exist_ok=True)
summaries = []

# 分别尝试两种策略，对比耗时、元素数量、分类和文本内容。
for strategy in ("hi_res", "ocr_only"):
    print(f"\n{'=' * 60}\n解析策略: {strategy}", flush=True)
    started = perf_counter()
    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy=strategy,
            languages=["chi_sim", "eng"],
        )
    except Exception as exc:
        # 一种策略失败后仍继续尝试另一种策略。
        print(f"{strategy} 解析失败: {type(exc).__name__}: {exc}", flush=True)
        summaries.append({"strategy": strategy, "error": f"{type(exc).__name__}: {exc}"})
        continue

    elapsed = perf_counter() - started
    print(f"耗时: {elapsed:.2f} 秒")
    print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")
    types = Counter(e.category for e in elements)
    print(f"元素类型: {dict(types)}")
    summaries.append({
        "strategy": strategy,
        "seconds": round(elapsed, 2),
        "elements": len(elements),
        "characters": sum(len(str(e)) for e in elements),
        "types": dict(types),
    })
    (output_dir / f"{strategy}.txt").write_text(
        "\n\n".join(
            f"[{i}] {e.category} | page={e.metadata.page_number}\n{e}"
            for i, e in enumerate(elements, 1)
        ),
        encoding="utf-8",
    )

    print("\n所有元素:")
    for i, element in enumerate(elements, 1):
        print(f"Element {i} ({element.category}):")
        print(element)
        print("=" * 60)

(output_dir / "comparison.json").write_text(
    json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
)
comparison = "\n\n".join(
    f"{s['strategy']}: " + (
        s["error"] if "error" in s else
        f"{s['seconds']} 秒，{s['elements']} 个元素，{s['characters']} 字符\n"
        f"元素类型: {s['types']}"
    )
    for s in summaries
)
(output_dir / "comparison.txt").write_text(comparison, encoding="utf-8")
print(f"\n策略对比:\n{comparison}", flush=True)
print(f"\n结果目录: {output_dir}", flush=True)
if any("error" in summary for summary in summaries):
    raise SystemExit(1)
