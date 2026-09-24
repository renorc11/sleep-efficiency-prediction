import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Sleep Efficiency Predictor",
  description: "Lifestyle + screen inactivity based interpretable sleep prediction",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
