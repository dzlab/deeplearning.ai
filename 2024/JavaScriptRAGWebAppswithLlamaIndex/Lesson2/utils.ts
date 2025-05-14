let rootDir = Deno.cwd()

const runFrontend = () => {
    Deno.chdir("frontend")

    let npmInstall = new Deno.Command(
      "npm", {
          args: ["ci"]
      }).outputSync()

    let result = new Deno.Command(
        "npm",{
            args: ["run","dev"]
        }).output()
	//    return new TextDecoder().decode(result.stdout)
	return {
	  [Symbol.for("Jupyter.display")]() {
	      return {
	          "text/plan": Deno.env.get("DLAI_LOCAL_URL").replace(/\{\}/, Deno.env.get("PORT1")), 
		  "text/html": "Check <a href=" + Deno.env.get("DLAI_LOCAL_URL").replace(/\{\}/, Deno.env.get("PORT1")) + " target='_blank'>"+ Deno.env.get("DLAI_LOCAL_URL").replace(/\{\}/, Deno.env.get("PORT1")) + "</a> to see the results" 
              }
	  }
	}
}


const runBackend = () => {
    Deno.chdir(rootDir)
    let result = new Deno.Command(
        "deno",{
            args: ["run","--allow-read","--allow-env","--allow-net","server.ts"]
        }).output()
    return new TextDecoder().decode(result.stdout)
}

const addToFrontend = async (cellNumber) => {
    let result = await new Deno.Command(
        "node",{
            args: ["./inject-frontend.mjs",cellNumber]
        }).output()    
}

const runCreateLlamaApp = async() => {
    Deno.chdir("l4-app/frontend")
    let run1 = new Deno.Command(
        "npm",{
            args: ["run","dev"]
        }).output()
    Deno.chdir("../backend")
    let run2 = new Deno.Command(
        "npm",{
            args: ["run","dev"]
        }).output()
}

export {
    runFrontend,
    runBackend,
    addToFrontend,
    runCreateLlamaApp
}
