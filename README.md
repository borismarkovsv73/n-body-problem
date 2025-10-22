# Projekat NTP – N-body problem  
**Varijanta za ocenu: 6**

---

## Opis problema  
N-body problem predstavlja klasičan problem u oblasti simulacija i numeričke matematike, gde se modelira kretanje više tela koja međusobno deluju gravitacionom silom.  
Svako telo ima svoju masu, položaj i brzinu, a njegova putanja kroz prostor zavisi od rezultujuće sile koja na njega deluje, tj. od interakcije sa svim ostalim telima u sistemu.  

Problem je važan jer se koristi za modelovanje kretanja planeta, zvezda, galaksija, ali i kao primer u oblasti visokih performansi zbog potrebe za velikim brojem računanja.  

---

## Metod rešavanja  

1. **Sekvencijalna verzija (Python)**  
   - U svakoj iteraciji izračunavaju se gravitacione sile koje deluju između svih parova tela.  
   - Na osnovu dobijene rezultujuće sile za svako telo ažuriraju se brzine i pozicije.  
   - Promene stanja (položaji i brzine tela po iteracijama) biće sačuvane u datoteci.

2. **Paralelizovana verzija (Python, `multiprocessing`)**  
   - Računanje gravitacionih sila između tela može se paralelizovati jer su međusobno nezavisne po parovima tela.  
   - Upotrebom `multiprocessing` biblioteke zadatak će biti podeljen između više procesa, čime se postiže ubrzanje simulacije.  
   - Kao rezultat, dobiće se ista izlazna datoteka sa stanjem sistema po iteracijama, ali brže generisana u odnosu na sekvencijalnu verziju.  

---

## Implementacija  

Simulacija je implementirana kroz klasu `NBodySimulator` koja enkapsulira kompletan sistem za modelovanje n-body problema. Svako telo je predstavljeno preko `Body` klase koja sadrži masu, poziciju i brzinu.

**Komponente:**
- **Algoritam integracije:** Koristi se Euler metod za numeričku integraciju Njutnovih jednačina kretanja
- **Paralelizacija:** Računanje sila podeljeno je u chunks između worker procesa pomoću `multiprocessing.Pool`
- **Visualizacija:** Tri tipa prikaza - statički grafici, 2D animacija i 3D animacija sa trail efektima
- **Eksport podataka:** Čuvanje kompletnog stanja simulacije u JSON formatu za dalju analizu

## Pokretanje

1. Instaliranje zavnosti: `pip install -r requirements_simple.txt`
2. Pokretanje: `python nbody.py` 
3. Praćenje interaktivnih opcija za broj tela, iteracija i tip vizuelizacije