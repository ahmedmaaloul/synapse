import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Allow backend API calls during SSR
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
