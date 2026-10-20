import { Fragment } from "react";

/**
 * A file path that prefers to wrap after a slash and breaks inside a segment only when that segment alone is too
 * wide. With `range` (e.g. "31–39"), the ":31–39" part never breaks, so a line number is never split.
 */
export function FilePath({ path, range }: { path: string; range?: string }) {
  const parts = path.split("/");
  const name = parts.pop()!;
  return (
    <>
      {parts.map((dir, i) => (
        <Fragment key={i}>
          {dir}/<wbr />
        </Fragment>
      ))}
      {name}
      {range && <span className="whitespace-nowrap">:{range}</span>}
    </>
  );
}
