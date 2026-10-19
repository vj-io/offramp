import { Suspense } from "react";
import { LiveRun } from "./live-run";

export default function RunPage() {
  return (
    <Suspense fallback={<p className="font-hand text-xl text-muted">Opening the run…</p>}>
      <LiveRun />
    </Suspense>
  );
}
