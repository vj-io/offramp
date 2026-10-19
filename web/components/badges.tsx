// Hand-lettered sticker badges, text only (no copied logos).
export function BrandBadges({ small = false }: { small?: boolean }) {
  const size = small ? "px-1.5 py-0 text-[0.8rem]" : "px-2.5 py-0.5 text-base";
  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      <span className={`sketch bg-note-plain font-hand leading-snug text-ink ${size}`} style={{ rotate: "-1.5deg" }}>
        Nebius Token Factory
      </span>
      <span className={`sketch bg-note-plain font-hand leading-snug text-ink ${size}`} style={{ rotate: "1.2deg" }}>
        NVIDIA Nemotron
      </span>
    </span>
  );
}
