FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL https://claude.ai/install.sh | sh && \
    apt-get purge -y --auto-remove curl && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r cascade && \
    useradd -r -g cascade -m -s /bin/bash cascade

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER cascade
WORKDIR /workspace

ENTRYPOINT ["entrypoint.sh"]
