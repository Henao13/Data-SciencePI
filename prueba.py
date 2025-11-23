from app.model_service import predict_single

"""ejemplo = {
    "Operario": "Edward Dario Pulgarin Paniagua",
    "NombreMaquina": "Bordadora 12",
    "codigoMaterial": "U53554",

    "Puntadas": 1735.0,
    "CantidadBuena": 162,
    "PuntoPlantilla": "1",

    "FechaInicio": "2021-06-15 11:21:39.9748690",

    # tiempos asociados al material (los puedes mandar)
}
print(predict_single(ejemplo))"""

import json
from app.lambda_function import handler

ejemplo = {
    "Operario": "Edward Dario Pulgarin Paniagua",
    "NombreMaquina": "Bordadora 12",
    "codigoMaterial": "U53554",
    "Puntadas": 1735.0,
    "CantidadBuena": 162,
    "PuntoPlantilla": "1",
    "FechaInicio": "2021-06-15 11:21:39.9748690",
}

event = {
    "body": json.dumps(ejemplo)  # así lo manda API Gateway normalmente
}

resp = handler(event, None)
print(resp)
print("Body:", resp["body"])
