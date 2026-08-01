const vscode = require('vscode');
const net = require('net');

function getConfig() {
    return vscode.workspace.getConfiguration('debugpySpawn');
}

const attachedPorts = new Set();
let server = null;
let ipcPort = 0;

async function attachToWorker(port, name) {
    if (attachedPorts.has(port)) return;
    attachedPorts.add(port);
    const config = getConfig();
    try {
        await vscode.debug.startDebugging(undefined, {
            type: "debugpy",
            request: "attach",
            name: `${name}`,
            connect: { host: config.get('host'), port },
            justMyCode: config.get('justMyCode')
        });
    } catch (e) {
        attachedPorts.delete(port);
    }
}

function activate(context) {
    console.log('debugpy-spawn: activated');

    // 1. Start the IPC Server on a dynamic port
    server = net.createServer((socket) => {
        socket.on('data', (data) => {
            const message = data.toString().trim();
            const [portStr, name] = message.split(':');
            const port = parseInt(portStr, 10);
            if (!isNaN(port) && name) {
                attachToWorker(port, name);
            }
        });
    });

    // Listen on port 0 to let the OS assign a free port
    server.listen(0, '127.0.0.1', () => {
        ipcPort = server.address().port;
        console.log(`debugpy-spawn: IPC server listening on port ${ipcPort}`);
    });

    // 2. Inject the dynamic port into the Python environment
    context.subscriptions.push(
        vscode.debug.registerDebugConfigurationProvider('debugpy', {
            resolveDebugConfiguration(folder, config) {
                if (config.request === 'launch') {
                    if (!config.env) {
                        config.env = {};
                    }
                    if (ipcPort > 0) {
                        config.env._DEBUG_SPAWN_EXT_PORT = ipcPort.toString();
                    }
                }
                return config;
            }
        })
    );

    context.subscriptions.push(
        vscode.debug.onDidTerminateDebugSession(() => {
            attachedPorts.clear();
        })
    );
}

function deactivate() {
    if (server) {
        server.close();
    }
}

module.exports = { activate, deactivate };