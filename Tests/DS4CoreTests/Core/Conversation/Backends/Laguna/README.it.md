[English](README.md) | **Italiano**

# Test conversazione Laguna

Golden test contro il framing del transcript di riferimento `laguna-s2.1`:
l'apertura `〈|EOS|〉`, il blocco system con la sezione `### Tools` /
`<available_tools>`, i turni user/tool-response con tag testuali, i turni
assistant con reasoning interlacciato e chiusura `</assistant>`, le chiamate
ai tool taggate senza separatori, il parser strict/streaming dei tool, il
contenimento del contenuto non fidato e i default di sampling di riferimento.
