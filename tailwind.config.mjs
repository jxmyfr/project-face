/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
      fontFamily: {
        beiruti: ['Beiruti', 'sans-serif'],
        // ต้องใส่ "sans-serif" ให้ครบ
        bebasneue: ["Bebas Neue", "sans-serif"],
      },
    },
  },
  plugins: [],
}