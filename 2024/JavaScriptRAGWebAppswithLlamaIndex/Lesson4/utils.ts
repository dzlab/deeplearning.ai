let rootDir = Deno.cwd()

const runFrontend = () => {
    Deno.chdir("frontend")

    let result = new Deno.Command(
        "npm",{
            args: ["run","dev"]
        }).output()
    return new TextDecoder().decode(result.stdout)
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

const runCreateLlamaApp = () => {
    Deno.chdir(rootDir + "/l4-app/frontend")
    //Deno.chdir("l4-app/frontend")
    let npmFrontendInstall = new Deno.Command(
      "npm", {
          args: ["ci"]
      }).outputSync()

    let run1 = new Deno.Command(
        "npm",{
            args: ["run","dev"]
        }).output()
    Deno.chdir(rootDir + "/l4-app/backend")
    //Deno.chdir("../backend")

    let npmBackendInstall = new Deno.Command(
      "npm", {
          args: ["ci"]
      }).outputSync()
    let run2 = new Deno.Command(
        "npm",{
            args: ["run","dev"]
        }).output()

    return {
          [Symbol.for("Jupyter.display")]() {
              return {
                  "text/plan": Deno.env.get("DLAI_LOCAL_URL").replace(/\{\}/, Deno.env.get("PORT1")),
		  "text/html": "Check <a href=" + Deno.env.get("DLAI_LOCAL_URL").replace(/\{\}/, Deno.env.get("PORT1")) + " target='_blank'>"+ Deno.env.get("DLAI_LOCAL_URL").replace(/\{\}/, Deno.env.get("PORT1")) + "</a> you'll see a web app set up with our data! "
              }
          }
        }

}

export {
    runFrontend,
    runBackend,
    addToFrontend,
    runCreateLlamaApp
}
