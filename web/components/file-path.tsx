import { Fragment } from "react";

/**
 * A file path that prefers to wrap after a slash. With `range` (e.g. "31–39"), the last segment and the line range
 * stay together as one unit that only breaks as a last resort, when it alone is wider than its line.
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
      {range ? (
        <span className="inline-block max-w-full">
          {name}:{range}
        </span>
      ) : (
        name
      )}
    </>
  );
}
