import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import { AuthProvider } from '@/context/AuthContext';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'SureshGPT — Phase 3 ChatGPT-Style AI Platform',
  description: 'Enterprise AI chat platform powered by Groq Llama 3.3 70B & Qwen 2.5 with document RAG knowledge base',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark h-full">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased h-full bg-[#121212] text-zinc-100 flex flex-col`}
      >
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
