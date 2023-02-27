FROM node

WORKDIR /app
COPY package.json .
RUN npm install --no-cache
COPY . .

EXPOSE 5173
CMD ["npm", "run", "dev"]