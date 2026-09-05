FROM node:20-alpine AS runner

WORKDIR /app

# Set environment
ENV NODE_ENV=development
ENV NEXT_TELEMETRY_DISABLED=1

# Copy package manifests and install
COPY package.json ./
RUN npm install

# Copy application source
COPY . .

# Expose Next.js port
EXPOSE 3000

CMD ["npm", "run", "dev"]
