/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    async rewrites() {
      return [
        {
          source: '/api/:path*',
          destination: 'https://casca-loan-evaluation.onrender.com/:path*' // Use your actual Render URL 
        }
      ];
    },
};

module.exports = nextConfig;
