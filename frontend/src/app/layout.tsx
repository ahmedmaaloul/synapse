import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Project Synapse — Knowledge Explorer",
  description:
    "Interactive GraphRAG Knowledge Explorer. Upload documents, visualize knowledge graphs, and chat with your data.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
