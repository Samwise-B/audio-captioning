FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg

# Install readline and bash completion tools
RUN apt-get update && apt-get install -y \
    readline-common \
    bash-completion \
    rlwrap \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# The bash-completion package installs its scripts, but you might need to manually source them in your .bashrc or /etc/bash.bashrc.
# Add this snippet to your Dockerfile to enable completion globally:
RUN echo "if [ -f /usr/share/bash-completion/bash_completion ]; then \
    . /usr/share/bash-completion/bash_completion; \
fi" >> /etc/bash.bashrc

# Keep container running
CMD ["/bin/bash", "-1"]