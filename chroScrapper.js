const cheerio = require('cheerio');
const axios = require('axios')
const fs = require('fs')

const  OUTPUT_FILE = 'humanChronology.txt'
const BASE_URL = `https://fr.wikipedia.org/wiki` 
const listOfSource  = [
    `Chronologie_de_la_France_sous_Louis_XIV`,
    `Chronologie_de_la_Révolution_française`,
    `Chronologie_de_la_Révolution_française_et_du_Premier_Empire`,
    `Chronologie_de_la_France_sous_le_premier_Empire`
]




async  function getContent(url) {
    console.log( `Parsing url ${url}`)
    const resp = await axios.get(url)
    const $ = cheerio.load(resp.data)


    const eventList  = $('li')

    for (let i=0; i<eventList.length; i++) {
        const el = eventList[i]
        let sequence = $(el).text()
        sequence += '\n'
        fs.appendFile(OUTPUT_FILE, sequence, function(err, data) {
            console.error('==============================')
            console.log(err)
        })
    }

}


listOfSource.forEach( myUrl => getContent(`${BASE_URL}/${myUrl}`))
