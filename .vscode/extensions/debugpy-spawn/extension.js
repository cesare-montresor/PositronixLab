const vscode = require('vscode');

let _pattern = null;

function getConfig() {
    return vscode.workspace.getConfiguration('debugpySpawn');
}

function getPattern() {
    if (!_pattern) {
        _pattern = new RegExp(getConfig().get('portPattern'));
    }
    return _pattern;
}

const attachedPorts = new Set();

async function attachToWorker(port) {
    if (attachedPorts.has(port)) return;
    attachedPorts.add(port);

    const config = getConfig();
    vscode.window.showInformationMessage(`debugpy-spawn: worker detected on port ${port}, attaching...`);
    try {
        await vscode.debug.startDebugging(undefined, {
            type: "debugpy",
            request: "attach",
            name: `Worker :${port}`,
            connect: { host: config.get('host'), port },
            justMyCode: config.get('justMyCode')
        });
    } catch (e) {
        // port gone — remove so it can retry if worker restarts
        attachedPorts.delete(port);
    }
}

function activate(context) {
    console.log('debugpy-spawn: activated');

    context.subscriptions.push(
        vscode.debug.registerDebugConfigurationProvider('debugpy', {
            resolveDebugConfiguration(folder, config) {
                if (config.request === 'launch') {
                    config.redirectOutput = true;
                }
                return config;
            }
        })
    );

    // only start watching terminal when a debug session starts
    context.subscriptions.push(
        vscode.debug.onDidStartDebugSession(session => {
            if (session.type !== 'debugpy') return;
            console.log('debugpy-spawn: debug session started, watching terminal');

            const listener = vscode.window.onDidWriteTerminalData(e => {
                const match = e.data.match(getPattern());
                if (match) attachToWorker(parseInt(match[1]));
            });

            // stop watching when session ends
            vscode.debug.onDidTerminateDebugSession(terminated => {
                if (terminated.id === session.id) {
                    listener.dispose();
                    attachedPorts.clear();
                }
            });
        })
    );
}



function deactivate() {}
module.exports = { activate, deactivate };