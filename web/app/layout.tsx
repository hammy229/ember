import type { Metadata, Viewport } from "next";
import "@fontsource/spectral/300.css";
import "@fontsource/spectral/400.css";
import "@fontsource-variable/inter-tight/index.css";
import "@fontsource-variable/jetbrains-mono/index.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ember",
  description: "An LLM inference engine written from scratch in C++, running in your browser.",
};

export const viewport: Viewport = {
  themeColor: "#15110e",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
