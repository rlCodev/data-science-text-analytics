# FROM node as build

# WORKDIR /app
# COPY package.json .
# RUN npm install --no-cache
# COPY . .
# RUN npm run build

# FROM nginx:1.23.1-alpine
# EXPOSE 80
# COPY ./nginx/conf.d/default.conf /etc/nginx/conf.d/default.conf
# COPY --from=build /usr/app/dist /usr/share/nginx/html

FROM node

WORKDIR /app
COPY package.json .
RUN npm install --no-cache
COPY . .
## EXPOSE [Port you mentioned in the vite.config file]
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]