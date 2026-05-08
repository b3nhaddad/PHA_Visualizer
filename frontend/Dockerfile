FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

# BACKEND_URL is injected by docker-compose so the Vite proxy knows where to forward /api
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
