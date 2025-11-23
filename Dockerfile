FROM public.ecr.aws/lambda/python:3.12

# 1. Copiar requirements e instalarlos
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2. Copiar el código de la app y los modelos
COPY app/ ./app
COPY models/ ./models

# 3. Asegurar que Python vea el paquete "app"
ENV PYTHONPATH="/var/task"

# 4. Comando de arranque: archivo.handler
CMD ["app.lambda_function.handler"]
