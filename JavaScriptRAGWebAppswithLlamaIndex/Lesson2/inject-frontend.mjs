import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from 'node:url';

const injection_path = "/frontend/src/app/my-component.tsx"

// read in the notebook
const cwd = path.dirname(fileURLToPath(import.meta.url))
const data = fs.readFileSync(path.join(cwd, 'L2_Build_a_full-stack_web_app.ipynb'), 'utf8')
const cells = JSON.parse(data).cells

// which cell are we looking for?
let cellToFind = process.argv[2]
let searchString = `// import-me: ${cellToFind}`

// search through the cells for the one with our searchstring
const findCode = (cells, searchString) => {
    for(let cell of cells) {
        if(cell.cell_type == "code") {
            if(cell.source[0]?.includes(searchString)) {
                // found it!
                return cell.source
            }
        }
    }
}

// format the code 
let codeLines = findCode(cells, searchString)
// make it a client component
codeLines.unshift(`"use client";\n`)
// change the imports to Node from Deno
for (let line in codeLines) {
    if (codeLines[line].includes("import React")) {
        codeLines[line] = `import React, { useState } from 'react';`
    }
    if (codeLines[line].includes("http://localhost:8000")) {
	codeLines[line] = codeLines[line].replace(/http\:\/\/localhost\:8000/, process.env.DLAI_LOCAL_URL.replace(/\{\}/, process.env.PORT2))
    }
}
// convert to a string and write to file at injection_path
let codeString = codeLines.join("")
fs.writeFileSync(path.join(cwd,injection_path), codeString)
