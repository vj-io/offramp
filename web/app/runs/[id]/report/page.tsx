import type { Metadata } from "next";
import { Suspense } from "react";
import { ReportView } from "./report-view";

export const metadata: Metadata = { title: "Run report" };

export default function ReportPage() {
  return (
    <Suspense fallback={<p className="font-hand text-xl text-muted">Opening the report…</p>}>
      <ReportView />
    </Suspense>
  );
}
