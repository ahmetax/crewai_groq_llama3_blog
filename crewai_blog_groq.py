import os
import sys
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool

def get_myapikey(key):
    with open("/etc/apikeys/myapikeys.py") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('GROQ_API_KEY'):
                s = line.split('=')
                skey=s[1].strip()
                skey=skey.replace("'","")
                return skey

GROQ_API_KEY=get_myapikey('GROQ_API_KEY')

os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] = 'llama3-70b-8192'
os.environ["OPENAI_API_KEY"] = GROQ_API_KEY

konu = "Bilim kurgu filmleri"

researcher = Agent(
    role='Uzman araştırmacı',
    goal='Verilen konuya uygun konu başlıkları oluştur ve bu başlıkları ayrıntılı bir şekilde araştır.',
    backstory="""
    Sen uzman bir araştırmacısın. 
    Araştırma yaparken İngilizce kullanıyorsun.
    Teknik konulara hakimsin.""",
    verbose=True,
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

result = crew.kickoff()

print(result)
