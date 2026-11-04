import type { NextConfig } from "next";

// A static site: the engine runs in the visitor's browser, so there is no server to pay for.
const nextConfig: NextConfig = {
  output: "export",
};

export default nextConfig;
