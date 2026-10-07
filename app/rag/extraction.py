


from pydantic import Field
import os
from typing import Optional

from dotenv import load_dotenv
from groq import BaseModel, Groq
import instructor

load_dotenv()

client=instructor.from_groq(
   Groq(api_key=os.getenv("GROQ_API_KEY")),
   mode= instructor.Mode.TOOLS,
)

class FinancialMetrics(BaseModel):

    company_name: Optional[str]= Field(None,
                     description="The name of the company " 
                    "this data refers to, e.g. 'Tesla, Inc.'")
    fiscal_year: Optional[int]= Field(None,
                    description="The fiscal year this data refers to" 
                    ", e.g. 2023")
    revenue_usd_millions: Optional[float]=Field(None,
                    description="Total revenue in millions of USD, Convert \
                    billions if needed ( e.g. $96.77 billion ->96770)."
    )

    net_income_usd_millions: Optional[float]=Field(
        None,description="Net income in millions of USD. Can be negative if the company had a net loss."
    )
    rd_spend_usd_millions: Optional[float] = Field(
        None, description="Research and development spending in millions of USD."
    )
    gross_profit_margin_percent: Optional[float] = Field(
        None, description="Gross profit margin as a percentage, e.g. 19.4 for 19.4%."
    )
    operating_income_usd_millions: Optional[float] = Field(
        None, description="Operating income in millions of USD."
    )
    cash_and_equivalents_usd_millions: Optional[float] = Field(
        None, description="Cash and cash equivalents in millions of USD, typically from the balance sheet."
    )
 
    source_quote: Optional[str] = Field(
        None, description="A short direct quote (under 20 words) from the context that supports the extracted numbers, for verification purposes."
    )
    not_found_in_context: bool = Field(
        False, description="Set to true ONLY if none of the requested financial data could be found in the provided context."
    )

def extract_financial_data(context:str)->FinancialMetrics:

    result=client.chat.completions.create(
        model="openai/gpt-oss-120b",
        response_model=FinancialMetrics,
        max_retries=2,
        messages=[
            {
                "role":"system",
                "content":( 
                    "you are a financial data extraction assistant.extract only the "
                    "financial metrics in the context provided."
                    "do not guess or infer values that are not stated . leave fields empty "
                    "if the information is not present. "

                ),
                
            },
            {
                "role":"user",
                "content":f"Context:\n{context}\n\nExtract the financial metrics from this context.",

            },
        ],
    )
    return result
    

