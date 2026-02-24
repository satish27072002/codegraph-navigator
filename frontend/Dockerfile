FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY . .

# Allow VITE_API_BASE to be injected at build time via docker-compose build.args
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=$VITE_API_BASE

RUN npm run build

EXPOSE 3000
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "3000"]
