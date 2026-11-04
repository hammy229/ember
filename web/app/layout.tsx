import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ember",
  description: "An LLM inference engine written from scratch in C++, running in your browser.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
