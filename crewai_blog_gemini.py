import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import tool

def show_time(message, t1):
    t2 = time.time()
    print(f"{message}: {round(t2-t1,4)} saniye")
    return t2

def get_myapikey(key):
    with open("/etc/apikeys/myapikeys.py") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(key):
                s = line.split('=')
                skey=s[1].strip()
                skey=skey.replace("'","")
                return skey

GOOGLE_API_KEY=get_myapikey('GOOGLE_API_KEY')

@tool('DuckDuckGoSearch')
def search(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)

# Set gemini pro as llm
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose = True,
                             temperature = 0.3,
                             google_api_key=GOOGLE_API_KEY)

konu = "Bilim kurgu filmleri"

researcher = Agent(
    role='Uzman araştırmacı',
    goal='Verilen konuya uygun konu başlıkları oluştur ve bu başlıkları ayrıntılı bir şekilde araştır.',
    backstory="""
    Sen uzman bir araştırmacısın. 
    Araştırma yaparken İngilizce kullanıyorsun.
    Teknik konulara hakimsin.""",
    verbose=True,
    llm=llm,
    allow_delegation=False
)

writer = Agent(
    role='Blog yazarı',
    goal='Edindiğin bilgileri kullanarak çarpıcı, bilgilendirici ve uzman diliyle yazılmış Türkçe yazı üret. ',
    backstory="""
    Sen uzman bir blog yazarısın. 
    Teknik konulara hakimsin. 
    Teknik ve bilgilendirici içerik üretmek senin uzmanlık alanın.
    Olabildiğince kısa ve anlaşılır cümleler kuruyorsun.
    Eğer kullanacağın bilgilerde eksikler varsa, researcher'a soruyorsun.
    """,
    verbose=True,
    llm=llm,
    allow_delegation=True   # Eksik bilgileri araştırmacıya sorabilmek için
)

research_task = Task(
    description=f"""{konu} hakkında araştırma yap""",
    expected_output='',
    agent=researcher,
    verbose=True
)

write_task = Task(
    description=f"""{konu} hakkında uygun uzunlukta Türkçe bir makale yaz. 
    Yazı dikkat çekici olsun. 
    Magazin dili kullan.""",
    expected_output='',
    agent=writer,
    verbose=True
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=2,
    process=Process.sequential
)

t1 = time.time()
result = crew.kickoff()
print(result)
show_time("Toplam süre ", t1)
