import fs from "node:fs/promises";
import {
  SpreadsheetFile,
  Workbook,
} from "file:///C:/Users/Javi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const resultDir = process.argv[2];
if (!resultDir) throw new Error("Usage: node phase4_build_review_workbook.mjs <result-dir>");

async function readTsv(name) {
  const text = (await fs.readFile(`${resultDir}/${name}`, "utf8")).trim();
  return text.split(/\r?\n/).map((line) => line.split("\t").map((value) => {
    if (value === "") return null;
    if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(value)) return Number(value);
    return value;
  }));
}

function colName(index) {
  let value = index + 1;
  let out = "";
  while (value > 0) {
    value -= 1;
    out = String.fromCharCode(65 + (value % 26)) + out;
    value = Math.floor(value / 26);
  }
  return out;
}

function styleTable(sheet, data, widths = []) {
  const rows = data.length;
  const cols = data[0].length;
  const end = colName(cols - 1);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.getRange(`A1:${end}${rows}`).values = data;
  const header = sheet.getRange(`A1:${end}1`);
  header.format.fill = "#1F4E78";
  header.format.font = { bold: true, color: "#FFFFFF" };
  header.format.wrapText = true;
  header.format.rowHeight = 34;
  const body = sheet.getRange(`A2:${end}${rows}`);
  body.format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  body.format.rowHeight = 20;
  for (let i = 0; i < cols; i += 1) {
    sheet.getRange(`${colName(i)}:${colName(i)}`).format.columnWidth = widths[i] || 18;
  }
  return { rows, cols, end };
}

const workbook = Workbook.create();
const readme = workbook.worksheets.add("Read Me");
const summarySheet = workbook.worksheets.add("Summary");

const sheetSpecs = [
  ["Trait Counts", "before_after_counts.tsv", [30, 18, 15, 20, 20, 20, 20]],
  ["Reliability", "trait_reliability_summary.tsv", [30, 12, 24, 22, 22, 24]],
  ["Model Selection", "model_selection_by_trait.tsv", [30, 18, 20, 22, 25, 15]],
  ["Ranking Ceiling", "ranking_ceiling_by_trait.tsv", [30, 12, 24, 25, 20, 28, 22]],
  ["Unreliable", "unreliable_by_trait.tsv", [30, 12, 20, 20]],
  ["Check Status", "check_status_summary.tsv", [40, 24, 18]],
  ["Robust", "robust_sensitivity_summary.tsv", [42, 14, 72]],
  ["Validation", "validation_checks.tsv", [42, 12, 44, 44]],
];

const sheetData = {};
for (const [sheetName, filename, widths] of sheetSpecs) {
  const data = await readTsv(filename);
  sheetData[sheetName] = data;
  const sheet = workbook.worksheets.add(sheetName);
  const meta = styleTable(sheet, data, widths);
  if (["Reliability", "Ranking Ceiling", "Unreliable"].includes(sheetName)) {
    sheet.getRange(`C2:${meta.end}${meta.rows}`).format.numberFormat = "0.000";
  }
  if (sheetName === "Validation") {
    sheet.getRange(`B2:B${meta.rows}`).conditionalFormats.add("containsText", {
      text: "PASS",
      format: { fill: "#E2F0D9", font: { color: "#375623", bold: true } },
    });
  }
}

readme.showGridLines = false;
readme.getRange("A1:H2").merge();
readme.getRange("A1:H2").values = [["Phase 4 phenotype reconstruction — review workbook"]];
readme.getRange("A1:H2").format.fill = "#17365D";
readme.getRange("A1:H2").format.font = { bold: true, color: "#FFFFFF", size: 18 };
readme.getRange("A1:H2").format.verticalAlignment = "center";
readme.getRange("A4:B12").values = [
  ["Release", "phase4_phenotype_reconstruction_2026_08_01_v1"],
  ["Status", "PASS — independently validated 19/19"],
  ["Scope", "Seven predeclared modelling traits; 37,206 exact environment/trait/original-trait/unit groups"],
  ["Protected outcomes", "Outer-test outcomes and final holdout were not opened or used"],
  ["Spatial constraint", "No independent field row/column; AR1×AR1 is not identifiable"],
  ["Recommended target", "Within-group selected adjusted BLUE with PEV proxy and reliability"],
  ["Deregression", "Not needed for the recommended BLUE; needed if BLUP is substituted"],
  ["Outliers", "No observations deleted; robust weights and flags are diagnostic only"],
  ["Source of truth", "Validated Parquet/TSV files in the same Phase-4 release directory"],
];
readme.getRange("A4:A12").format.fill = "#D9EAF7";
readme.getRange("A4:A12").format.font = { bold: true, color: "#17365D" };
readme.getRange("A4:B12").format.borders = { preset: "all", style: "thin", color: "#B4C6E7" };
readme.getRange("A:A").format.columnWidth = 24;
readme.getRange("B:B").format.columnWidth = 95;
readme.getRange("B4:B12").format.wrapText = true;
readme.getRange("A4:B12").format.rowHeight = 30;

summarySheet.showGridLines = false;
summarySheet.getRange("A1:F2").merge();
summarySheet.getRange("A1:F2").values = [["Phase 4 validated outcomes"]];
summarySheet.getRange("A1:F2").format.fill = "#1F4E78";
summarySheet.getRange("A1:F2").format.font = { bold: true, color: "#FFFFFF", size: 18 };
summarySheet.getRange("A4:B12").values = [
  ["Metric", "Value"],
  ["Canonical plot records", null],
  ["Adjusted phenotype entries", null],
  ["Trial–trait groups", null],
  ["Groups with estimable H²", 31376],
  ["Groups with ranking ceiling", null],
  ["Groups too unreliable for ranking", null],
  ["Observations excluded", 0],
  ["Independent validation gates", 19],
];
summarySheet.getRange("B5").formulas = [["=SUM('Trait Counts'!B2:B8)"]];
summarySheet.getRange("B6").formulas = [["=SUM('Trait Counts'!E2:E8)"]];
summarySheet.getRange("B7").formulas = [["=SUM('Model Selection'!F2:F8)"]];
summarySheet.getRange("B9").formulas = [["=SUM('Ranking Ceiling'!C2:C8)"]];
summarySheet.getRange("B10").formulas = [["=SUM('Unreliable'!C2:C8)"]];
summarySheet.getRange("A4:B4").format.fill = "#5B9BD5";
summarySheet.getRange("A4:B4").format.font = { bold: true, color: "#FFFFFF" };
summarySheet.getRange("A5:A12").format.fill = "#D9EAF7";
summarySheet.getRange("A5:A12").format.font = { bold: true, color: "#17365D" };
summarySheet.getRange("A4:B12").format.borders = { preset: "all", style: "thin", color: "#B4C6E7" };
summarySheet.getRange("B5:B12").format.numberFormat = "#,##0";
summarySheet.getRange("A:A").format.columnWidth = 38;
summarySheet.getRange("B:B").format.columnWidth = 20;
summarySheet.getRange("D4:F4").merge();
summarySheet.getRange("D4:F4").values = [["Interpretation"]];
summarySheet.getRange("D4:F4").format.fill = "#5B9BD5";
summarySheet.getRange("D4:F4").format.font = { bold: true, color: "#FFFFFF" };
summarySheet.getRange("D5:F10").merge(true);
summarySheet.getRange("D5:F10").values = [
  ["Adjustment preserves most entry ranking: median raw-vs-adjusted Spearman is ≥0.996 for all traits."],
  ["Median adjusted ceiling is highest for heading (0.875) and lowest for biomass (0.512)."],
  ["13,628 groups are explicitly too unreliable for ranking claims; no group is silently deleted."],
  ["35,564 groups select unadjusted means; 1,288 rep/block, 177 spline, and 177 plot-order AR1."],
  ["AR1×AR1 is not identifiable because independent row/column coordinates are absent."],
  ["BLUE is recommended. BLUP is included for sensitivity and would require deregression downstream."],
];
summarySheet.getRange("D5:F10").format.wrapText = true;
summarySheet.getRange("D5:F10").format.rowHeight = 42;
summarySheet.getRange("D5:F10").format.fill = "#F3F6FA";
summarySheet.getRange("D5:F10").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
summarySheet.getRange("D:F").format.columnWidth = 24;
summarySheet.freezePanes.freezeRows(2);

const inspect = await workbook.inspect({
  kind: "workbook,sheet,formula",
  maxChars: 12000,
  tableMaxRows: 12,
  tableMaxCols: 8,
});
await fs.writeFile(`${resultDir}/phase4_review_workbook_inspect.txt`, String(inspect.ndjson || inspect), "utf8");
const preview = await workbook.render({ sheetName: "Summary", autoCrop: "all", scale: 1.5, format: "png" });
await fs.writeFile(`${resultDir}/phase4_review_workbook_preview.png`, new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${resultDir}/phase4_review_workbook.xlsx`);
