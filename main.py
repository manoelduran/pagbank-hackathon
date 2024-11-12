from fastapi import FastAPI
from statsmodels.tsa.statespace.sarimax import SARIMAX
import boto3
import json
import pandas as pd
 
 
app = FastAPI()
 
 
# Inicializa o cliente boto3 para o bedrock
brt = boto3.client(service_name='bedrock-runtime')
 
 
# FUnção para aplicar SARIMA e gerar previsões de vendas acima e abaixo da média
def apply_sarima(personas_data):
    """ 
    Aplica o modelo SARIMA a uma série temporal de vendas e retorna previsões.
    """
    # COnfigura o modelo SARIMA 
    model = SARIMAX(personas_data, order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_fit = model.fit(disp=False)

    # Gerar previsões
    forecast = sarima_fit.predict(start=len(personas_data), end=len(personas_data) + 5)
    # Calcular a média de vendas da série
    avg_sales = personas_data.mean()

    return forecast, avg_sales

# Função para gerar respostas do modelo Bedrock para cada cliente no dicionário

def generate_responses(personas_data, investment_data):
    """ 
    FUnção para gerar respostas do modelo Bedrock para cada cliente no hash map.
    """
    # Construir o dicionário para armazenar os dados do cliente
    customer_data_map = {}
    for _, row in personas_data.iterrows():
        customer_id = row['idt_customer']
        sales_sum = row['sum']
        date_month = row['date_month']
    # Se o cliente já está no dicionário, adiciona os valores ao array existente
    if customer_id in customer_data_map:
        customer_data_map[customer_id]['sums'].append(sales_sum)
        customer_data_map[customer_id]['dates'].append(date_month)
    else:
    # Caso contrário, inicializa um novo registro para o cliente    
        customer_data_map[customer_id] = {
            'sums': [sales_sum],
            'dates': [date_month]
        }
# Selecionando um produto de investimento para oferta
cdb_product = investment_data.iloc[0]['produto']
cdb_percentual_cdi = investment_data.iloc[0]['percentual_cdi']
# Lista para armazenar todas as respostas
responses = []
# Itera sobre cada cliente no dicionário e gera um prompt individual
for customer_id, data in customer_data_map.items():
    # Preparar série temporal para SARIMA
    sales_data = pd.Series(data['sums'], index=pd.to_datetime(data['dates'])).asfreq('MS')
    # Aplicar o modelo SARIMA e calcular a média de vendas
    forecast, avg_sales = apply_sarima(sales_data)
    # Gerar o prompt para o cliente específico
    prompt = f"""
        Human: Para o cliente com ID {customer_id}, a média histórica de vendas é de {avg_sales} e a lista de previsões futuras é ${forecast}.
        - Se as {forecast} estiverem **acima da {avg_sales}**, ofereça o produto de investimento {cdb_product} com {cdb_percentual_cdi}% do CDI.
        - Se as {forecast} estiverem **abaixo da {avg_sales}**, ofereça uma linha de crédito proporcional às vendas desse cliente.

        Assistant:
     """
    # Preparar o corpo da requisição para o modelo
    body = json.dumps({
        "prompt": prompt.strip(),
        "max_tokens_to_sample": 300,
        "temperature": 0.1,
        "top_p": 0.9
    })
    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'

    # Invocar o modelo
    response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

    # Parsear a resposta
    response_body = json.loads(response.get('body').read())
    completion_text = response_body.get('completion')

    # Armazenar a resposta com o ID do cliente
    responses.append({"idt_customer": customer_id, "completion": completion_text})
# Retorna todas as respostas
return responses

# Função para ler os novos arquivos CSV com os dados de personas e investimentos
def read_csv_files():
    # Lendo o CSV de personas
    seller_personas_df = pd.read_csv("seller_personas_detail.csv")
    # Lendo o CSV de produtos de investimento
    cdb_investment_db = pd.read_csv("cdb_investment_product_list.csv")
    # Retornando os DataFrames para serem usados em outros lugares
    return seller_personas_df, cdb_investment_df

# Testando a leitura dos novos arquivos CSV
personas_data, investment_data = read_csv_files()
responses = generate_responses(personas_data, investment_data)

# Imprime todas as respostas geradas para cada cliente
for response in responses:
    print(f"RESPOSTA: {response['idt_customer']}: {response['completion']}")