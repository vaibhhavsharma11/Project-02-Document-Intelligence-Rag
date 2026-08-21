const state = {
    documents: [],
};


const elements = {
    fileInput: document.getElementById("file-input"),
    uploadButton: document.getElementById("upload-button"),
    uploadStatus: document.getElementById("upload-status"),

    documentList: document.getElementById("document-list"),
    documentCount: document.getElementById("document-count"),

    askQuery: document.getElementById("ask-query"),
    askButton: document.getElementById("ask-button"),
    askLoading: document.getElementById("ask-loading"),
    askResult: document.getElementById("ask-result"),
    answerContent: document.getElementById("answer-content"),
    askEvidence: document.getElementById("ask-evidence"),
    askResultCount: document.getElementById("ask-result-count"),

    compareQuery: document.getElementById("compare-query"),
    compareButton: document.getElementById("compare-button"),
    compareLoading: document.getElementById("compare-loading"),
    compareResult: document.getElementById("compare-result"),
    comparisonContent: document.getElementById(
        "comparison-content"
    ),

    documentA: document.getElementById("document-a"),
    documentB: document.getElementById("document-b"),

    documentAEvidence: document.getElementById(
        "document-a-evidence"
    ),

    documentBEvidence: document.getElementById(
        "document-b-evidence"
    ),

    documentACount: document.getElementById(
        "document-a-count"
    ),

    documentBCount: document.getElementById(
        "document-b-count"
    ),

    errorMessage: document.getElementById(
        "error-message"
    ),
};


function escapeHtml(value) {
    const div = document.createElement("div");

    div.textContent = value ?? "";

    return div.innerHTML;
}


function formatDocumentId(documentId) {
    if (!documentId) {
        return "Unknown document";
    }

    if (documentId.length <= 18) {
        return documentId;
    }

    return (
        documentId.slice(0, 8) +
        "…" +
        documentId.slice(-6)
    );
}


function showError(message) {
    elements.errorMessage.textContent =
        message || "Something went wrong.";

    elements.errorMessage.classList.remove(
        "hidden"
    );
}


function clearError() {
    elements.errorMessage.textContent = "";

    elements.errorMessage.classList.add(
        "hidden"
    );
}


function setLoading(
    element,
    visible
) {
    element.classList.toggle(
        "hidden",
        !visible
    );
}


function setButtonLoading(
    button,
    loading
) {
    button.disabled = loading;
}


async function parseResponse(response) {
    const contentType =
        response.headers.get(
            "content-type"
        ) || "";

    if (
        contentType.includes(
            "application/json"
        )
    ) {
        return response.json();
    }

    const text =
        await response.text();

    return {
        detail:
            text ||
            "The server returned an unexpected response.",
    };
}


function renderDocuments() {
    const documents =
        state.documents;

    elements.documentCount.textContent =
        documents.length;


    if (!documents.length) {

        elements.documentList.innerHTML = `
            <div class="empty-state">

                <div class="empty-icon">
                    +
                </div>

                <p>
                    No indexed documents
                </p>

                <span>
                    Upload a PDF to build your knowledge base.
                </span>

            </div>
        `;

    } else {

        elements.documentList.innerHTML =
            documents
                .map(
                    (document) => `
                        <div class="document-item">

                            <div class="document-item-top">

                                <div class="document-icon">
                                    PDF
                                </div>

                                <div>

                                    <div class="document-name">
                                        ${escapeHtml(
                                            document.document_id
                                        )}
                                    </div>

                                    <div class="document-id">
                                        ${escapeHtml(
                                            formatDocumentId(
                                                document.document_id
                                            )
                                        )}
                                    </div>

                                </div>

                            </div>


                            <div class="document-meta">

                                <span class="meta-pill">
                                    ${Number(
                                        document.chunk_count || 0
                                    )} chunks
                                </span>

                                <span class="meta-pill">
                                    ${Number(
                                        document.page_count || 0
                                    )} pages
                                </span>

                                <span class="meta-pill">
                                    ${Number(
                                        document.character_count || 0
                                    )} chars
                                </span>

                            </div>

                        </div>
                    `
                )
                .join("");
    }


    populateDocumentSelectors();
}


function populateDocumentSelectors() {

    const currentA =
        elements.documentA.value;

    const currentB =
        elements.documentB.value;


    const options =
        state.documents
            .map(
                (document) => `
                    <option
                        value="${escapeHtml(
                            document.document_id
                        )}"
                    >
                        ${escapeHtml(
                            formatDocumentId(
                                document.document_id
                            )
                        )}
                    </option>
                `
            )
            .join("");


    elements.documentA.innerHTML = `
        <option value="">
            Select a document
        </option>
        ${options}
    `;


    elements.documentB.innerHTML = `
        <option value="">
            Select a document
        </option>
        ${options}
    `;


    if (
        state.documents.some(
            (document) =>
                document.document_id ===
                currentA
        )
    ) {
        elements.documentA.value =
            currentA;
    }


    if (
        state.documents.some(
            (document) =>
                document.document_id ===
                currentB
        )
    ) {
        elements.documentB.value =
            currentB;
    }
}


async function loadDocuments() {

    try {

        const response =
            await fetch(
                "/documents"
            );


        const data =
            await parseResponse(
                response
            );


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Unable to load documents."
            );
        }


        state.documents =
            data.documents || [];


        renderDocuments();

    } catch (error) {

        showError(
            error.message ||
            "Unable to load documents."
        );
    }
}


function renderEvidence(
    container,
    results
) {

    if (
        !results ||
        !results.length
    ) {

        container.innerHTML = `
            <div class="empty-state">

                <div class="empty-icon">
                    —
                </div>

                <p>
                    No evidence retrieved
                </p>

                <span>
                    No matching source was returned.
                </span>

            </div>
        `;

        return;
    }


    container.innerHTML =
        results
            .map(
                (result, index) => {

                    const metadata =
                        result.metadata || {};


                    const relevance =
                        typeof result.relevance_score ===
                        "number"
                            ? `${(
                                result.relevance_score *
                                100
                            ).toFixed(1)}%`
                            : "—";


                    return `
                        <article class="evidence-item">

                            <div class="evidence-top">

                                <div>

                                    <div class="source-label">
                                        Source ${index + 1}
                                    </div>

                                    <div class="evidence-meta">

                                        <span>
                                            Document:
                                            ${escapeHtml(
                                                formatDocumentId(
                                                    metadata.document_id
                                                )
                                            )}
                                        </span>

                                        <span>
                                            ·
                                        </span>

                                        <span>
                                            Page:
                                            ${escapeHtml(
                                                String(
                                                    metadata.page_number ??
                                                    "—"
                                                )
                                            )}
                                        </span>

                                        <span>
                                            ·
                                        </span>

                                        <span>
                                            Chunk:
                                            ${escapeHtml(
                                                String(
                                                    metadata.chunk_index ??
                                                    "—"
                                                )
                                            )}
                                        </span>

                                    </div>

                                </div>


                                <div class="relevance">
                                    ${relevance}
                                    relevance
                                </div>

                            </div>


                            <div class="evidence-text">
                                ${escapeHtml(
                                    result.text || ""
                                )}
                            </div>

                        </article>
                    `;
                }
            )
            .join("");
}


async function askDocuments() {

    clearError();


    const query =
        elements.askQuery.value.trim();


    if (!query) {

        showError(
            "Please enter a question."
        );

        elements.askQuery.focus();

        return;
    }


    setLoading(
        elements.askLoading,
        true
    );


    elements.askResult.classList.add(
        "hidden"
    );


    setButtonLoading(
        elements.askButton,
        true
    );


    try {

        const response =
            await fetch(
                "/documents/ask",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",
                    },

                    body: JSON.stringify({
                        query,

                        top_k: 3,

                        distance_threshold: 450,
                    }),
                }
            );


        const data =
            await parseResponse(
                response
            );


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Question answering failed."
            );
        }


        elements.answerContent.textContent =
            data.answer || "";


        const results =
            data.results || [];


        elements.askResultCount.textContent =
            `${results.length} ${
                results.length === 1
                    ? "source"
                    : "sources"
            }`;


        renderEvidence(
            elements.askEvidence,
            results
        );


        elements.askResult.classList.remove(
            "hidden"
        );


        requestAnimationFrame(
            () => {
                elements.askResult.scrollIntoView(
                    {
                        behavior: "smooth",
                        block: "start",
                    }
                );
            }
        );

    } catch (error) {

        showError(
            error.message ||
            "Question answering failed."
        );

    } finally {

        setLoading(
            elements.askLoading,
            false
        );

        setButtonLoading(
            elements.askButton,
            false
        );
    }
}


async function compareDocuments() {

    clearError();


    const documentA =
        elements.documentA.value;


    const documentB =
        elements.documentB.value;


    const query =
        elements.compareQuery.value.trim();


    if (!documentA || !documentB) {

        showError(
            "Please select both documents."
        );

        return;
    }


    if (documentA === documentB) {

        showError(
            "Document A and Document B must be different."
        );

        return;
    }


    if (!query) {

        showError(
            "Please enter a comparison question."
        );

        elements.compareQuery.focus();

        return;
    }


    setLoading(
        elements.compareLoading,
        true
    );


    elements.compareResult.classList.add(
        "hidden"
    );


    setButtonLoading(
        elements.compareButton,
        true
    );


    try {

        const response =
            await fetch(
                "/documents/compare",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",
                    },

                    body: JSON.stringify({
                        document_a_id:
                            documentA,

                        document_b_id:
                            documentB,

                        query,

                        top_k: 2,
                    }),
                }
            );


        const data =
            await parseResponse(
                response
            );


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Document comparison failed."
            );
        }


        elements.comparisonContent.textContent =
            data.answer || "";


        const resultsA =
            data.document_a_results || [];


        const resultsB =
            data.document_b_results || [];


        elements.documentACount.textContent =
            `${resultsA.length} ${
                resultsA.length === 1
                    ? "source"
                    : "sources"
            }`;


        elements.documentBCount.textContent =
            `${resultsB.length} ${
                resultsB.length === 1
                    ? "source"
                    : "sources"
            }`;


        renderEvidence(
            elements.documentAEvidence,
            resultsA
        );


        renderEvidence(
            elements.documentBEvidence,
            resultsB
        );


        elements.compareResult.classList.remove(
            "hidden"
        );


        requestAnimationFrame(
            () => {
                elements.compareResult.scrollIntoView(
                    {
                        behavior: "smooth",
                        block: "start",
                    }
                );
            }
        );

    } catch (error) {

        showError(
            error.message ||
            "Document comparison failed."
        );

    } finally {

        setLoading(
            elements.compareLoading,
            false
        );

        setButtonLoading(
            elements.compareButton,
            false
        );
    }
}


async function uploadDocument(file) {

    if (!file) {
        return;
    }


    clearError();


    if (
        !file.name
            .toLowerCase()
            .endsWith(".pdf")
    ) {

        showError(
            "Only PDF files are supported."
        );

        return;
    }


    elements.uploadStatus.textContent =
        "Uploading and indexing…";


    setButtonLoading(
        elements.uploadButton,
        true
    );


    try {

        const formData =
            new FormData();


        formData.append(
            "file",
            file
        );


        const response =
            await fetch(
                "/documents/upload",
                {
                    method: "POST",

                    body: formData,
                }
            );


        const data =
            await parseResponse(
                response
            );


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Document upload failed."
            );
        }


        elements.uploadStatus.textContent =
            `Indexed ${
                Number(
                    data.chunk_count || 0
                )
            } chunks successfully.`;


        await loadDocuments();


    } catch (error) {

        elements.uploadStatus.textContent =
            "";

        showError(
            error.message ||
            "Document upload failed."
        );

    } finally {

        setButtonLoading(
            elements.uploadButton,
            false
        );

        elements.fileInput.value = "";
    }
}


function setupTabs() {

    const tabs =
        document.querySelectorAll(
            ".mode-tab"
        );


    tabs.forEach(
        (tab) => {

            tab.addEventListener(
                "click",
                () => {

                    const mode =
                        tab.dataset.mode;


                    tabs.forEach(
                        (item) =>
                            item.classList.toggle(
                                "active",
                                item === tab
                            )
                    );


                    document
                        .getElementById(
                            "ask-panel"
                        )
                        .classList.toggle(
                            "active",
                            mode === "ask"
                        );


                    document
                        .getElementById(
                            "compare-panel"
                        )
                        .classList.toggle(
                            "active",
                            mode === "compare"
                        );


                    clearError();
                }
            );
        }
    );
}


function setupEvents() {

    setupTabs();


    elements.uploadButton.addEventListener(
        "click",
        () => {
            elements.fileInput.click();
        }
    );


    elements.fileInput.addEventListener(
        "change",
        () => {

            const file =
                elements.fileInput.files[0];


            uploadDocument(file);
        }
    );


    elements.askButton.addEventListener(
        "click",
        askDocuments
    );


    elements.compareButton.addEventListener(
        "click",
        compareDocuments
    );


    elements.askQuery.addEventListener(
        "keydown",
        (event) => {

            if (
                event.key === "Enter" &&
                (event.metaKey ||
                    event.ctrlKey)
            ) {

                event.preventDefault();

                askDocuments();
            }
        }
    );


    elements.compareQuery.addEventListener(
        "keydown",
        (event) => {

            if (
                event.key === "Enter" &&
                (event.metaKey ||
                    event.ctrlKey)
            ) {

                event.preventDefault();

                compareDocuments();
            }
        }
    );
}


async function initialize() {

    setupEvents();

    await loadDocuments();
}


initialize();