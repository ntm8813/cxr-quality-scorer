from __future__ import annotations
from pathlib import Path
from src.reports.pdf_report_generator import generate_batch_report


def main() -> None:
    Path("reports/pdf").mkdir(parents=True, exist_ok=True)

    output = generate_batch_report(
        predictions_csv="data/predictions/model_v1.csv",
        kappa_json="reports/interrater_kappa.json",
        calibration_png="reports/figures/validation_calibration.png",
        failure_md="reports/failure_catalogue.md",
        output_path="reports/pdf/batch_qa_report.pdf",
    )

    print(f"Generated: {output}")


if __name__ == "__main__":
    main()