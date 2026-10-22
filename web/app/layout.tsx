import type { Metadata } from "next";
import Link from "next/link";
import { Archivo, JetBrains_Mono, Patrick_Hand } from "next/font/google";
import { BrandBadges } from "@/components/badges";
import { ThemeToggle } from "@/components/theme-toggle";
import { THEME_KEY } from "@/lib/theme";
import "./globals.css";

const archivo = Archivo({ subsets: ["latin"], variable: "--ff-archivo" });
const hand = Patrick_Hand({ subsets: ["latin"], weight: "400", variable: "--ff-hand" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--ff-mono" });

const description =
  "Shows, on held-out cases, which closed-LLM call sites NVIDIA Nemotron on Nebius Token Factory can take over, and opens the pull request.";

export const metadata: Metadata = {
  metadataBase: new URL("https://offramp-seven.vercel.app"),
  title: { default: "Offramp", template: "%s · Offramp" },
  description,
  openGraph: { siteName: "Offramp", title: "Offramp", description, type: "website" },
  twitter: { card: "summary_large_image", title: "Offramp", description },
};

// Applies a saved paper/blackboard choice before first paint.
const themeScript = `try{var t=localStorage.getItem("${THEME_KEY}");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning className={`${archivo.variable} ${hand.variable} ${mono.variable} antialiased`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-dvh">
        <header className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-6 gap-y-3 px-4 pt-5 pb-2 sm:px-6">
          <Link href="/" className="group flex items-baseline gap-2 rounded-sm">
            <span className="text-3xl font-extrabold tracking-tight">Offramp</span>
            <svg aria-hidden="true" viewBox="0 0 60 30" className="h-6 w-12 overflow-visible">
              <path d="M2 26 L34 26 Q46 26 52 8" className="ink-stroke" strokeWidth="3" />
              <path d="M44 9 L52 6 L55 14" className="ink-stroke" strokeWidth="3" />
            </svg>
          </Link>
          <div className="flex flex-wrap items-center gap-3">
            <BrandBadges />
            <ThemeToggle />
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 pt-4 pb-16 sm:px-6">{children}</main>
        <footer className="mx-auto flex max-w-6xl flex-wrap gap-x-5 gap-y-1 px-4 pb-8 font-hand text-lg text-muted sm:px-6">
          <a href="https://github.com/vj-io/offramp" className="text-bluepen underline underline-offset-4">
            Source code
          </a>
          <a
            href="https://github.com/vj-io/offramp/blob/main/docs/FEEDBACK.md"
            className="text-bluepen underline underline-offset-4"
          >
            Feedback on Token Factory and Nemotron
          </a>
          {/* The demo video's link goes here once it is uploaded. */}
        </footer>
      </body>
    </html>
  );
}
