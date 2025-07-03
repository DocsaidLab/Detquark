# syntax=docker/dockerfile:experimental
FROM nvcr.io/nvidia/pytorch:25.05-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONWARNINGS="ignore" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    tzdata wget git libturbojpeg exiftool ffmpeg poppler-utils libpng-dev \
    libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev gcc \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    python3-pip libharfbuzz-dev libfribidi-dev libxcb1-dev libfftw3-dev \
    libpq-dev python3-dev gosu && \
    ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir -U pip setuptools wheel

RUN python -m pip install --no-cache-dir -U \
    tqdm colored ipython tabulate tensorboard scikit-learn fire \
    albumentations "Pillow>=10.0.0" fitsne opencv-fixer prettytable \
    ftfy regex tqdm ultralytics lightning

COPY ./Capybara /usr/local/Capybara
RUN cd /usr/local/Capybara && \
    python setup.py bdist_wheel && \
    python -m pip install dist/*.whl && \
    cd ~ && rm -rf /usr/local/Capybara

COPY ./Chameleon /usr/local/Chameleon
RUN cd /usr/local/Chameleon && \
    python setup.py bdist_wheel && \
    python -m pip install dist/*.whl && \
    cd ~ && rm -rf /usr/local/Chameleon

RUN python -c "from opencv_fixer import AutoFix; AutoFix()"
RUN python -c "import capybara; import chameleon"

WORKDIR /code

ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

RUN printf '#!/bin/bash\n\
    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
    groupadd -g "$GROUP_ID" -o usergroup\n\
    useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
    export HOME=/home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /code\n\
    exec gosu "$USER_ID":"$GROUP_ID" "$@"\n\
    else\n\
    exec "$@"\n\
    fi' > "$ENTRYPOINT_SCRIPT" && \
    chmod +x "$ENTRYPOINT_SCRIPT"

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]

CMD ["bash"]
