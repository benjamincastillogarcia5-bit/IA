import * as vscode from 'vscode';
import axios from 'axios';

export function activate(context: vscode.ExtensionContext) {
    console.log('Extensión "rnn-autocompleter" activada.');

    const provider: vscode.InlineCompletionItemProvider = {
        async provideInlineCompletionItems(document, position, context, token) {
            const startPosition = position.character > 40
                ? new vscode.Position(position.line, position.character - 40)
                : new vscode.Position(position.line, 0);

            const textContext = document.getText(new vscode.Range(startPosition, position));

            // AÑADE ESTA LÍNEA PARA DEPURAR:
            console.log("Intentando autocompletar. Texto leído:", textContext);

            if (textContext.trim().length === 0) { return []; }
            try {
                // Consumir nuestra API local en FastAPI
                const response = await axios.post('http://127.0.0.1:8000/predict', {
                    context: textContext
                });

                const suggestion = response.data.suggestion;

                // Retornar la sugerencia para que aparezca "en fantasma" (ghost text)
                return [
                    new vscode.InlineCompletionItem(suggestion)
                ];
            } catch (error) {
                console.console.error();
                ("Error contactando a la RNN:", error);
                return [];
            }
        }
    };

    // Registrar para archivos en lenguaje C
    const disposable = vscode.languages.registerInlineCompletionItemProvider(
        { language: 'c' },
        provider
    );

    context.subscriptions.push(disposable);
}

export function deactivate() { }
