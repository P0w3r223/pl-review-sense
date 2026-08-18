"""The 80 base probe sentences. Written by us; no PolEmo text appears here.

Labels are the reading a Polish speaker gives the sentence, and each cell is balanced so
that no single answer wins it:

* ``plain`` — 7 negative, 6 neutral, 7 positive
* ``negation`` — 8 negative, 4 neutral, 8 positive, every one carrying a negation cue
* ``sarcasm`` — 10 ironic (negative) against 10 genuine (positive) in the same register
* ``contrast`` — 8 negative, 4 neutral, 8 positive, sentiment landing after the pivot

The domains echo PolEmo's — hotels, medicine, products, school — so the probe asks about
phenomena rather than about vocabulary the model has never seen.
"""

from __future__ import annotations

from .challenge import CONTRAST, NEGATION, PLAIN, SARCASM, Case

NEGATIVE, NEUTRAL, POSITIVE = 0, 1, 2

CASES: tuple[Case, ...] = (
    # --- plain: the control cell -------------------------------------------------------
    Case("Fatalna jakość wykonania, obudowa pękła po dwóch dniach.", NEGATIVE, PLAIN),
    Case("Obsługa w recepcji była opryskliwa i lekceważąca.", NEGATIVE, PLAIN),
    Case("Lekarz spóźnił się godzinę, a cała wizyta trwała cztery minuty.", NEGATIVE, PLAIN),
    Case("Zamówienie dotarło uszkodzone, a reklamację odrzucono.", NEGATIVE, PLAIN),
    Case("W pokoju czuć było wilgoć i stęchliznę.", NEGATIVE, PLAIN),
    Case("Zajęcia prowadzone chaotycznie, materiał kompletnie przestarzały.", NEGATIVE, PLAIN),
    Case("Telefon rozładowuje się w pół dnia i grzeje przy ładowaniu.", NEGATIVE, PLAIN),
    Case("Hotel leży dwa przystanki od dworca, śniadanie w formie bufetu.", NEUTRAL, PLAIN),
    Case("Wizyta trwała dwadzieścia minut, lekarz zlecił podstawowe badania krwi.", NEUTRAL, PLAIN),
    Case("Laptop ma matowy ekran i dwa porty USB-C, waży półtora kilograma.", NEUTRAL, PLAIN),
    Case("Kurs obejmuje dwanaście spotkań po dziewięćdziesiąt minut, materiały w PDF.", NEUTRAL, PLAIN),
    Case("Przesyłka przyszła kurierem w trzy dni robocze, opakowanie standardowe.", NEUTRAL, PLAIN),
    Case("Gabinet mieści się na parterze, rejestracja działa od ósmej do szesnastej.", NEUTRAL, PLAIN),
    Case("Świetny sprzęt, działa dokładnie tak, jak opisano.", POSITIVE, PLAIN),
    Case("Personel wyjątkowo pomocny, pokój sprzątany codziennie.", POSITIVE, PLAIN),
    Case("Pani doktor wytłumaczyła wszystko spokojnie i dokładnie.", POSITIVE, PLAIN),
    Case("Bateria trzyma dwa dni, a ekran jest czytelny w pełnym słońcu.", POSITIVE, PLAIN),
    Case("Wykładowca tłumaczy jasno i chętnie odpowiada na pytania.", POSITIVE, PLAIN),
    Case("Zamówienie dotarło następnego dnia, wszystko zapakowane wzorowo.", POSITIVE, PLAIN),
    Case("Jedzenie w restauracji hotelowej było naprawdę smaczne.", POSITIVE, PLAIN),

    # --- negation: the cue decides the direction ---------------------------------------
    Case("Nie polecam, sprzęt zepsuł się po miesiącu używania.", NEGATIVE, NEGATION),
    Case("Nie jest to produkt wart swojej ceny.", NEGATIVE, NEGATION),
    Case("Żaden z obiecanych zabiegów nie został wykonany.", NEGATIVE, NEGATION),
    Case("Pokój bez okna, bez klimatyzacji i bez możliwości wietrzenia.", NEGATIVE, NEGATION),
    Case("Nigdy więcej nie zapiszę się do tej przychodni.", NEGATIVE, NEGATION),
    Case("Brak jakiegokolwiek kontaktu z obsługą po zakupie.", NEGATIVE, NEGATION),
    Case("Ani jedne zajęcia nie odbyły się o zapowiedzianej godzinie.", NEGATIVE, NEGATION),
    Case("Nie da się tego używać bez ciągłego resetowania.", NEGATIVE, NEGATION),
    Case("Hotel nie ma własnego parkingu, najbliższy jest po drugiej stronie ulicy.", NEUTRAL, NEGATION),
    Case("Kurs nie obejmuje egzaminu, certyfikat wydawany jest za dopłatą.", NEUTRAL, NEGATION),
    Case("Model nie ma czytnika linii papilarnych, odblokowanie działa kodem.", NEUTRAL, NEGATION),
    Case("Przychodnia nie przyjmuje w soboty, rejestracja wyłącznie telefoniczna.", NEUTRAL, NEGATION),
    Case("Nie jest to zły sprzęt, spełnia wszystko, czego od niego oczekiwałem.", POSITIVE, NEGATION),
    Case("Nie znalazłem w tym hotelu niczego, do czego mógłbym się przyczepić.", POSITIVE, NEGATION),
    Case("Ani razu nie musiałem czekać dłużej niż pięć minut.", POSITIVE, NEGATION),
    Case("Nie spodziewałem się tak dobrej obsługi w tej cenie.", POSITIVE, NEGATION),
    Case("Żadna z moich obaw się nie potwierdziła, wszystko zadziałało od razu.", POSITIVE, NEGATION),
    Case("Nigdy wcześniej nie trafiłem na tak cierpliwego lekarza.", POSITIVE, NEGATION),
    Case("Nie mam żadnych zastrzeżeń, sprzęt działa bez zarzutu.", POSITIVE, NEGATION),
    Case("Bez najmniejszego problemu przeszedłem cały kurs, materiały są znakomite.", POSITIVE, NEGATION),

    # --- sarcasm: ironic praise, and its genuine twin ----------------------------------
    Case("Świetnie, kolejna rzecz, która rozpadła się po tygodniu.", NEGATIVE, SARCASM),
    Case("Rewelacyjna obsługa — czekałem tylko dwie godziny na odpowiedź.", NEGATIVE, SARCASM),
    Case("Genialny pomysł, żeby wyłączyć windę w ośmiopiętrowym hotelu.", NEGATIVE, SARCASM),
    Case("Cudownie, przesyłka dotarła miesiąc po terminie, brawo.", NEGATIVE, SARCASM),
    Case("Wspaniale, że regulamin zmienia się już po opłaceniu kursu.", NEGATIVE, SARCASM),
    Case("Piękna sprawa: gwarancja obejmuje wszystko poza tym, co się zepsuło.", NEGATIVE, SARCASM),
    Case("Fantastyczna organizacja, dwie wizyty zapisane na tę samą godzinę.", NEGATIVE, SARCASM),
    Case("No i pięknie, aplikacja zawiesza się dokładnie przy płatności.", NEGATIVE, SARCASM),
    Case("Super, że o odwołaniu zajęć dowiedziałem się dopiero na miejscu.", NEGATIVE, SARCASM),
    Case("Jestem zachwycony, bateria wytrzymała całe trzy godziny.", NEGATIVE, SARCASM),
    Case("Świetnie, że winda działa na każdym piętrze, bardzo to ułatwia pobyt.", POSITIVE, SARCASM),
    Case("Rewelacyjna obsługa — odpowiedź dostałem w dziesięć minut.", POSITIVE, SARCASM),
    Case("Genialny pomysł z automatycznym przypomnieniem o wizycie.", POSITIVE, SARCASM),
    Case("Cudownie, przesyłka dotarła dzień przed terminem.", POSITIVE, SARCASM),
    Case("Wspaniale, że regulamin kursu jest jasny od pierwszego dnia.", POSITIVE, SARCASM),
    Case("Piękna sprawa: gwarancja objęła naprawę bez żadnych pytań.", POSITIVE, SARCASM),
    Case("Fantastyczna organizacja, każda wizyta punktualnie o wyznaczonej godzinie.", POSITIVE, SARCASM),
    Case("No i pięknie, aplikacja przechodzi przez płatność w kilka sekund.", POSITIVE, SARCASM),
    Case("Super, że o zmianie terminu zajęć uprzedzono mnie z wyprzedzeniem.", POSITIVE, SARCASM),
    Case("Jestem zachwycony, bateria wytrzymała całe trzy dni.", POSITIVE, SARCASM),

    # --- contrast: the sentiment lands after the pivot ---------------------------------
    Case("Cena kusząca, ale jakość wykonania fatalna.", NEGATIVE, CONTRAST),
    Case("Personel miły, jednak pokój był brudny i głośny.", NEGATIVE, CONTRAST),
    Case("Sprzęt ładnie wygląda, ale przestał działać po dwóch tygodniach.", NEGATIVE, CONTRAST),
    Case("Chociaż lokalizacja jest świetna, hałas z ulicy uniemożliwia sen.", NEGATIVE, CONTRAST),
    Case("Materiały przygotowane porządnie, szkoda, że prowadzący zniknął w połowie kursu.", NEGATIVE, CONTRAST),
    Case("Mimo dobrych opinii wizyta okazała się stratą czasu i pieniędzy.", NEGATIVE, CONTRAST),
    Case("Zaczęło się obiecująco, skończyło na trzech reklamacjach.", NEGATIVE, CONTRAST),
    Case("Kupiłem z polecenia, ale żałuję każdej wydanej złotówki.", NEGATIVE, CONTRAST),
    Case("Pokój mniejszy niż na zdjęciach, ale łóżko wygodne — wychodzi na to samo.", NEUTRAL, CONTRAST),
    Case("Kurs ma słabsze momenty i mocne, trudno powiedzieć, czy bym go polecił.", NEUTRAL, CONTRAST),
    Case("Sprzęt ma lepszy ekran niż poprzednik, jednak gorszą baterię.", NEUTRAL, CONTRAST),
    Case("Wizyta droga, ale za to bez kolejki — plus za jedno, minus za drugie.", NEUTRAL, CONTRAST),
    Case("Dojazd męczący, ale sam pobyt wynagrodził wszystko.", POSITIVE, CONTRAST),
    Case("Początek był chaotyczny, jednak ostatecznie jestem bardzo zadowolony.", POSITIVE, CONTRAST),
    Case("Zapowiadało się przeciętnie, ale ostatecznie zmieniłem zdanie na tak.", POSITIVE, CONTRAST),
    Case("Cena wysoka, ale sprzęt wart każdej wydanej złotówki.", POSITIVE, CONTRAST),
    Case("Mimo drobnych usterek pobyt uważam za bardzo udany.", POSITIVE, CONTRAST),
    Case("Czekałem dłużej, niż zakładałem, jednak efekt przerósł oczekiwania.", POSITIVE, CONTRAST),
    Case("Pokój ciasny, ale obsługa i śniadania rekompensują to z nawiązką.", POSITIVE, CONTRAST),
    Case("Choć zajęcia zaczynały się wcześnie, kurs był najlepszy z dotychczasowych.", POSITIVE, CONTRAST),
)
