FROM node:20-slim

RUN npm install -g @openai/codex && \
    npm cache clean --force

RUN groupadd -r cascade && \
    useradd -r -g cascade -m -s /bin/bash cascade

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER cascade
WORKDIR /workspace

ENTRYPOINT ["entrypoint.sh"]
