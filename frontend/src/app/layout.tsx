import type { Metadata } from "next";
import { Inter, Space_Grotesk } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-space-grotesk",
  display: "swap",
});

const SITE_TITLE = "Synapse — GraphRAG Knowledge Explorer";
const SITE_DESCRIPTION =
  "Turn documents into a queryable Neo4j knowledge graph and chat with it using vector-grounded GraphRAG.";
const SITE_URL = "https://github.com/ahmedmaaloul/synapse";

export const metadata: Metadata = {
  title: SITE_TITLE,
  description: SITE_DESCRIPTION,
  applicationName: "Synapse",
  authors: [
    { name: "Ahmed Maaloul", url: "https://github.com/ahmedmaaloul" },
  ],
  creator: "Ahmed Maaloul",
  publisher: "Ahmed Maaloul",
  keywords: [
    "GraphRAG",
    "knowledge graph",
    "Neo4j",
    "RAG",
    "LLM",
    "Ahmed Maaloul",
  ],
  openGraph: {
    type: "website",
    url: SITE_URL,
    siteName: "Synapse",
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
  },
  twitter: {
    card: "summary_large_image",
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    creator: "@ahmedmaaloul",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${spaceGrotesk.variable}`}>
      <body>{children}</body>
    </html>
  );
}
