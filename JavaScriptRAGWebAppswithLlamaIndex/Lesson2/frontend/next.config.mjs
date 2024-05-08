/** @type {import('next').NextConfig} */
const nextConfig = {
    env: {
      API_URL: process.env.DLAI_LOCAL_URL,
      PORT: process.env.PORT

    }
};

export default nextConfig;
