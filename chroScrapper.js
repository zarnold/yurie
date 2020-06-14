const cheerio = require('cheerio');
const axios = require('axios')
const fs = require('fs')

const  OUTPUT_FILE = './DATA/humanChronology.txt'
const BASE_URL = `https://fr.wikipedia.org/wiki` 
const listOfSource  = [
    `Chronologie_de_la_France_sous_Louis_XIV`,
    `Chronologie_de_la_Révolution_française`,
    `Chronologie_de_la_Révolution_française_et_du_Premier_Empire`,
    `Chronologie_de_la_France_sous_le_premier_Empire`,
    `Chronologie_de_la_Grèce_antique`,
    `Chronologie_de_la_décolonisation`,
    `Chronologie_de_l%27histoire_des_techniques`,
    `Chronologie_de_la_France_rurale_(1848-1945)`,
    `Chronologie_des_faits_économiques`,
    `Chronologie_de_l%27esclavage`,
    `Chronologie_du_conflit_israélo-arabe`,
    `Chronologie_des_États-Unis_pendant_la_Seconde_Guerre_mondiale`,
    `Chronologie_de_Rome`,
    `Chronologie_de_la_République_romaine`,
    `Chronologie_de_la_première_croisade`,
    `Chronologie_de_l%27histoire_du_Québec`,
    `Chronologie_de_la_Révolution_française_et_du_Premier_Empire`,
    `Chronologie_de_l%27abolition_de_l%27esclavage`,
    `Chronologie_de_la_France_sous_le_premier_Empire`,
    `Chronologie_de_la_Commune_de_Paris`,
    `Chronologie_navale_de_la_Première_Guerre_mondiale`,
    `Chronologie_de_la_guerre_froide`,
    `Chronologie_de_la_décolonisation`,
    `Chronologie_navale_de_la_Première_Guerre_mondiale`
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
        fs.appendFile(OUTPUT_FILE, sequence,'utf8', function(err, data) {
            console.error('==============================')
            console.log(err)
        })
    }

}


listOfSource.forEach( myUrl => getContent(`${BASE_URL}/${myUrl}`))
