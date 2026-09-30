"""Build the experiment handoff XLSX/TSV from the launcher manifest (stdlib only)."""

import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "scripts/experiment_plans/linear_square_ablation/manifest.json"
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
HEADERS = [
    "标识号码", "类别", "Git分支", "评估指标", "方法说明", "配置（按实际条件修改）",
    "对应核心代码", "启动脚本", "保存名称", "文档Run", "安排", "启动命令",
    "保存目录（相对repo）", "日志目录（相对repo）", "代码状态", "实验状态", "优先级", "备注",
]
PRIORITY = {"04": "中", "08": "中", "11": "中", "12": "中"}
PURPOSES = {
    "01": "检验较小c；与Run05同c比较Linear和正负平方。",
    "02": "保留已有较强的Linear c15基准，作为固定c和后期衰减的对照。",
    "03": "与Run06同c比较Linear和正负平方。",
    "04": "把c15加倍，检验增幅收益是否饱和或更不稳定。",
    "05": "补足既有平方c5的观察时长；与Run01/09/11比较。",
    "06": "补足既有平方c20的观察时长，观察平台或回撤；与Run03/10/12比较。",
    "07": "与固定c15比较，检验后期适度减小目标修正能否缓解回撤。",
    "08": "检验更强衰减；与Run07和全程c5的Run01比较。",
    "09": "与Run01比较正侧平方；与Run05比较负侧保留原值还是平方。",
    "10": "在c20重复上述负侧比较，检查形状与c的耦合。",
    "11": "与Run09/05比较负侧均匀、原值、平方三种分配，正侧和c均相同。",
    "12": "在c20重复负侧三种分配比较，与Run10/06对照。",
}
METHODS = {
    "none": "Linear：f(r)=max(r,0)，组内归一化得Δw=w−1，正负shift保持原样，目标修正再乘c。",
    "square_both": "正侧按平方分配S+；负侧−min(1,τ|Δw|²)，保持原S−；映射后统一乘c。",
    "square_positive": "正侧按平方分配S+；负侧保留原Δw；映射后统一乘c。",
    "square_positive_uniform_negative": "正侧按平方分配S+；原负样本各取−S−/n−，不改变正负划分；映射后统一乘c。",
}


def experiment_rows(manifest, published):
    rows = []
    for run in manifest["runs"]:
        run_id, action = run["run_id"], run["action"]
        layout = "H200 8卡（单机）" if run["default_layout"] == "h200_8gpu" else "A6000 16卡（双机各8卡）"
        variant = run["layout_variants"][run["default_layout"]]
        scale = f"c={run['c']}"
        if run["schedule"] == "linear_decay":
            scale += f"→{run['target_scale_final']}；400–600步线性衰减，之后固定"
        config = "\n".join([
            f"默认{layout}；可通过--layout切换",
            "KL=1e-4; lr=3e-4; seed=42; fp16",
            "train/eval sample steps:10/40; timestep_fraction=1.0",
            "variance gate=false; 每卡batch=6",
            "48组×24图=1152图/轮；每轮1次更新尝试",
            f"rollout batches={variant['rollout_batches']}；梯度累积={variant['gradient_accumulation_steps']}",
            "总步数1000（global_step，含AMP跳过的尝试）；eval每10步/save每30步",
            scale,
        ])
        command = f"bash {run['script']}"
        notes = "同一预训练模型起跑；同配置已有完整checkpoint时自动续跑。"
        status, arrangement = "待运行", "新实验"
        code_status = "已发布" if published else "本地待审阅，尚未提交推送"
        if action == "resume":
            status, arrangement = "待恢复", "恢复已有B；计入Run02"
            notes = (
                "保留原c=15和历史保存目录；默认搜索完整checkpoint，找不到即报错，不从头跑。"
                "若原checkpoint不在默认目录，追加 --config.resume_from=/实际checkpoint或父目录。"
                "最近上传eval850；恢复步数以完整checkpoint为准。"
            )
        elif action == "continue":
            status, arrangement = "运行中（用户确认）", "继续现有任务，不重复启动"
            command = "无需新命令：保持当前任务运行。"
            code_status = "原分支已发布；现有任务继续运行"
            notes = (
                "最近上传eval270，不代表实时进度。原启动脚本未启用本次新增的末次eval功能。"
                "如任务中断，或需从完整checkpoint补到1000并保存/评估末步，"
                f"可在 {run['optional_resume_code_branch']} 发布后运行："
                f"bash {run['optional_resume_script']}。正在运行时勿重复启动。"
            )
        if action != "continue":
            notes += " 新入口完成训练后额外执行末次eval并保存完整checkpoint。"
        notes = "设计原因：" + PURPOSES[run_id] + "\n" + notes
        rows.append([
            run["sheet_id"], run["label"], run["code_branch"], "PickScore",
            METHODS[run["transform"]], config,
            "scripts/experiment_plans/step1/train_nft_sd3_ours-singleloss-AWR.py",
            run["script"], run["save_name"], run_id, arrangement, command,
            run["default_save_dir"], run["default_logdir"], code_status, status,
            PRIORITY.get(run_id, "高"), notes,
        ])
    return rows


def column_name(index):
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def xml_bytes(root):
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def worksheet(rows, widths, main=False):
    root = ET.Element("worksheet", xmlns=NS)
    ET.SubElement(root, "dimension", ref=f"A1:{column_name(len(rows[0]))}{len(rows)}")
    views = ET.SubElement(root, "sheetViews")
    view = ET.SubElement(views, "sheetView", workbookViewId="0")
    ET.SubElement(view, "pane", xSplit="2" if main else "1", ySplit="1",
                  topLeftCell="C2" if main else "B2", activePane="bottomRight", state="frozen")
    ET.SubElement(root, "sheetFormatPr", defaultRowHeight="18")
    cols = ET.SubElement(root, "cols")
    for index, width in enumerate(widths, 1):
        ET.SubElement(cols, "col", min=str(index), max=str(index), width=str(width), customWidth="1")
    data = ET.SubElement(root, "sheetData")
    for row_index, values in enumerate(rows, 1):
        style = 1 if row_index == 1 else 2
        if main and row_index > 1:
            style = 4 if values[0] == "B" else (5 if values[0] in ("C2.3", "C2.4") else 3)
        height = 32 if row_index == 1 else (210 if main else min(150, 32 + max(len(str(v)) // 70 for v in values) * 16))
        row = ET.SubElement(data, "row", r=str(row_index), ht=str(height), customHeight="1")
        for col_index, value in enumerate(values, 1):
            cell = ET.SubElement(row, "c", r=f"{column_name(col_index)}{row_index}", s=str(style), t="inlineStr")
            inline = ET.SubElement(cell, "is")
            text = ET.SubElement(inline, "t")
            text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            text.text = str(value)
    ET.SubElement(root, "autoFilter", ref=f"A1:{column_name(len(rows[0]))}{len(rows)}")
    ET.SubElement(root, "pageMargins", left="0.25", right="0.25", top="0.4", bottom="0.4", header="0.2", footer="0.2")
    ET.SubElement(root, "pageSetup", orientation="landscape", paperSize="9", fitToWidth="1", fitToHeight="0")
    return xml_bytes(root)


def styles():
    root = ET.Element("styleSheet", xmlns=NS)
    fonts = ET.SubElement(root, "fonts", count="2")
    for bold, color in ((False, "FF172B4D"), (True, "FFFFFFFF")):
        font = ET.SubElement(fonts, "font")
        ET.SubElement(font, "sz", val="10")
        ET.SubElement(font, "name", val="Microsoft YaHei")
        ET.SubElement(font, "color", rgb=color)
        if bold:
            ET.SubElement(font, "b")
    fills = ET.SubElement(root, "fills", count="6")
    for pattern, color in (("none", None), ("gray125", None), ("solid", "FF17365D"),
                           ("solid", "FFEAF4EA"), ("solid", "FFFFF2CC"), ("solid", "FFE6EFFA")):
        fill = ET.SubElement(fills, "fill")
        pattern_node = ET.SubElement(fill, "patternFill", patternType=pattern)
        if color:
            ET.SubElement(pattern_node, "fgColor", rgb=color)
            ET.SubElement(pattern_node, "bgColor", indexed="64")
    borders = ET.SubElement(root, "borders", count="1")
    border = ET.SubElement(borders, "border")
    for side in ("left", "right", "top", "bottom", "diagonal"):
        ET.SubElement(border, side)
    cell_styles = ET.SubElement(root, "cellStyleXfs", count="1")
    ET.SubElement(cell_styles, "xf", numFmtId="0", fontId="0", fillId="0", borderId="0")
    cell_xfs = ET.SubElement(root, "cellXfs", count="6")
    for index, (font, fill) in enumerate(((0, 0), (1, 2), (0, 0), (0, 3), (0, 4), (0, 5))):
        xf = ET.SubElement(cell_xfs, "xf", numFmtId="0", fontId=str(font), fillId=str(fill),
                           borderId="0", xfId="0", applyAlignment="1", applyFill="1", applyFont="1")
        ET.SubElement(xf, "alignment", vertical="center" if index == 1 else "top", wrapText="1")
    named_styles = ET.SubElement(root, "cellStyles", count="1")
    ET.SubElement(named_styles, "cellStyle", name="Normal", xfId="0", builtinId="0")
    return xml_bytes(root)


def write_xlsx(path, sheets):
    content = ET.Element("Types", xmlns="http://schemas.openxmlformats.org/package/2006/content-types")
    ET.SubElement(content, "Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
    ET.SubElement(content, "Default", Extension="xml", ContentType="application/xml")
    for name, content_type in (("/xl/workbook.xml", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"),
                                ("/xl/styles.xml", "application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml")):
        ET.SubElement(content, "Override", PartName=name, ContentType=content_type)
    package_rels = ET.Element("Relationships", xmlns=PACKAGE_REL_NS)
    ET.SubElement(package_rels, "Relationship", Id="rId1", Type=REL_NS + "/officeDocument", Target="xl/workbook.xml")
    workbook = ET.Element("workbook", xmlns=NS)
    ET.SubElement(workbook, "bookViews").append(ET.Element("workbookView", activeTab="0"))
    workbook_sheets = ET.SubElement(workbook, "sheets")
    workbook_rels = ET.Element("Relationships", xmlns=PACKAGE_REL_NS)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (name, rows, widths, is_main) in enumerate(sheets, 1):
            ET.SubElement(workbook_sheets, "sheet", name=name, sheetId=str(index),
                          **{f"{{{REL_NS}}}id": f"rId{index}"})
            ET.SubElement(workbook_rels, "Relationship", Id=f"rId{index}", Type=REL_NS + "/worksheet", Target=f"worksheets/sheet{index}.xml")
            ET.SubElement(content, "Override", PartName=f"/xl/worksheets/sheet{index}.xml",
                          ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml")
            archive.writestr(f"xl/worksheets/sheet{index}.xml", worksheet(rows, widths, is_main))
        ET.SubElement(workbook_rels, "Relationship", Id=f"rId{len(sheets) + 1}", Type=REL_NS + "/styles", Target="styles.xml")
        archive.writestr("[Content_Types].xml", xml_bytes(content))
        archive.writestr("_rels/.rels", xml_bytes(package_rels))
        archive.writestr("xl/workbook.xml", xml_bytes(workbook))
        archive.writestr("xl/_rels/workbook.xml.rels", xml_bytes(workbook_rels))
        archive.writestr("xl/styles.xml", styles())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/linear_square_ablation")
    parser.add_argument("--published", action="store_true", help="Use only after the delivery branch has actually been pushed.")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    assert len(manifest["runs"]) == 12
    assert {r["run_id"] for r in manifest["runs"]} == {f"{i:02}" for i in range(1, 13)}
    for run in manifest["runs"]:
        for key in ("script", "optional_resume_script"):
            if run.get(key) and not (ROOT / run[key]).is_file():
                raise ValueError(f"Missing launcher for {run['sheet_id']}: {run[key]}")
    rows = experiment_rows(manifest, args.published)
    release = "交付分支已推送；以发布时确认的commit为准。" if args.published else "本地待用户检查；新分支尚未提交/推送。请勿把本表当作远端代码已发布的通知。"
    instructions = [
        ["项目", "说明"],
        ["发布状态", release],
        ["代码分支", manifest["code_branch"]],
        ["本轮任务", "9组新实验 + 恢复B（Linear c15） + C2.3/C2.4保持运行，共12组。硬件差异不要求重复实验，但有效训练配置和每轮总图数保持一致。"],
        ["先检查配置", "新入口支持 --dry-run；只打印解析后的方法、参数和路径，不加载模型、不创建训练任务。"],
        ["直接运行", "从repo根目录执行Experiment Plan中的启动命令。每组方法、c、KL、seed等已经固定，不需要通过全局变量选择实验。"],
        ["硬件布局", "9组新实验统一A6000双机16卡，入口已默认此布局，无需传--layout；每卡batch=6，rollout batches和梯度累积均为12。B继续A6000；C2.3/C2.4保持原H200任务。仅需要另换资源时才显式传--layout；每轮总图数始终为48组×24图=1152。"],
        ["双机条件", "A6000双机仍需集群提供NODE_RANK、MASTER_ADDR等分布式连接信息；这是原有双机要求。单机8卡入口无需这些设置。"],
        ["模型和环境", "沿用现有DiffusionNFT conda环境、SD3.5 Medium权重、reward_ckpts和PickScore数据。模型路径等资源设置可按脚本帮助调整；不在本次交付中下载权重。"],
        ["B续跑", "默认在原B的保存目录查找最新完整checkpoint。若目录不同，追加 --config.resume_from=/实际checkpoint或父目录；没有完整checkpoint会报错，不会从头训练。"],
        ["现有两组平方", "C2.3/C2.4正在运行，保持运行，不同时启动第二份。任务中断或需补末次评估时，使用断点续跑子表中的可选入口。"],
        ["新实验起点", "9组新实验使用各自独立目录，从共同预训练模型开始；不从B后期checkpoint改变c或负侧映射开始。恢复只能接续同一配置。"],
        ["c衰减", "Run07/08前400步c=15，400–600步线性降至7.5/5，之后固定；按现有global_step尝试步计数，断点恢复会验证调度。"],
        ["负侧均匀", "仅在原负样本集合中均分原负总量−S−/n−，正侧继续平方；零shift不变，映射后再乘c，不新增最终系数截断。"],
        ["步数与末次评估", "新/续跑入口目标为总计1000步，不是再加1000步。每10步eval，训练结束额外eval并保存完整checkpoint；global_step包含AMP跳过的更新尝试。"],
        ["日志解释", "旧advantage_abs_mean保持乘c前定义；新增有效c和乘c后系数、target RMS、梯度裁剪和参数变化日志。correction_norm仍为均方，不能当作RMS或参数步幅。"],
        ["改配置与保存", "脚本固定实验参数，防止命令行或旧环境变量误选方法。不同实验不得共用保存目录。表中保存名按默认布局，切换布局以--dry-run输出为准。"],
        ["训练状态", "C2.3/C2.4的运行状态来自用户；270/850等历史步数是最近上传日志，不是实时进度，也不是可恢复checkpoint步数。"],
    ]
    resume_rows = [["编号", "何时使用", "Git分支", "启动命令", "默认checkpoint搜索目录", "说明"]]
    for run in manifest["runs"]:
        if run["action"] not in ("resume", "continue"):
            continue
        script = run.get("optional_resume_script", run["script"])
        resume_rows.append([
            run["sheet_id"], "B现在恢复" if run["action"] == "resume" else "原任务已停止时；勿并发重启",
            manifest["code_branch"], f"bash {script}", run["default_save_dir"],
            "require_resume=true；自动选择完整checkpoint并校验方法/c；若路径不同追加 --config.resume_from=/实际路径。",
        ])
    history_rows = [["原编号", "本轮安排", "原因"]]
    history_rows += [
        ["A1/A2", "不安排恢复或新启动", "旧Linear c1，保留历史记录。"],
        ["B", "恢复，计入Run02", "Linear c15核心配置一致，忽略卡差异，不重复新开同配置。"],
        ["C1/C2", "保持暂停/本轮不安排", "旧c1不等于本轮c5/20。"],
        ["C3/C4", "保持暂停", "本轮不做指数映射。"],
        ["C2.1/C2.2", "保持暂停/本轮不安排", "正负平方c10/15不属于当前12组。"],
        ["C2.3/C2.4", "继续运行", "对应Run05/06，不重复新开。"],
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "Linear_Square_Experiment_Handoff_20260929"
    tsv_path = args.output_dir / f"{stem}.tsv"
    with tsv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(HEADERS)
        writer.writerows(rows)
    xlsx_path = args.output_dir / f"{stem}.xlsx"
    write_xlsx(xlsx_path, [
        ("Experiment Plan", [HEADERS] + rows, [12, 32, 43, 13, 48, 65, 65, 70, 65, 12, 28, 78, 72, 72, 35, 23, 10, 90], True),
        ("运行说明", instructions, [24, 135], False),
        ("断点续跑", resume_rows, [12, 32, 43, 85, 82, 92], False),
        ("旧实验处理", history_rows, [23, 33, 95], False),
    ])
    print(xlsx_path)
    print(tsv_path)


if __name__ == "__main__":
    main()
