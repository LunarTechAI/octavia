FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy source
COPY . .

# Build the application
RUN npm run build

# Expose port
EXPOSE 3000

# Start application
CMD ["npm", "start"]
