/** @type {import('tailwindcss').Config} */
const colors = require('tailwindcss/colors')

export default {
  content: ["./src/**/*.{html,js,vue}"],
  theme: {
    colors: {
      black: colors.black,
      white: colors.white,
      gray: colors.gray,
      emerald: colors.emerald,
      indigo: colors.indigo,
      yellow: colors.yellow,
      'tw-yellow': '#ffca28',
      'tw-yellow-rbga': 'rgba(255, 202, 40, 0.5)',
      'tw-gray': 'rgba(255, 255, 255, 0.8)',
      'tw-dark-gray': '#232526',
    },
    extend: {},
  },
  plugins: [],
}

