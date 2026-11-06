struct JuliaHubCommercialSupportComponent <: MultiDocumenter.DropdownComponent
    link::String
end

function MultiDocumenter.render(c::JuliaHubCommercialSupportComponent, doc, thispage, prettyurls)

    return MultiDocumenter.@htl """
    <div class="sciml-commercial-support">
        <a href="https://juliahub.com/products/dyad" class="dyad-brand-card">
            <div class="dyad-brand-head">
                <img class="dyad-brand-logo" src="/assets/dyad-logo.svg" alt="" width="34" height="34"/>
                <div>
                    <span class="dyad-brand-title">Dyad</span>
                    <span class="dyad-brand-by">BY JULIAHUB</span>
                </div>
            </div>
            <div class="dyad-brand-body">
                <p class="dyad-brand-intro">Dyad by JuliaHub is the agentic modeling and simulation platform built on the SciML stack.</p>
                <ul class="dyad-brand-features">
                    <li>Component-based GUI for modeling</li>
                    <li>Modeling language and compiler stack integrated with and transparent to AI agents</li>
                </ul>
                <span class="dyad-brand-cta">Explore Dyad <span aria-hidden="true">↗</span></span>
            </div>
        </a>
        <p>JuliaHub offers commercial support for ModelingToolkit and the SciML ecosystem.
        <a href="$(c.link)" class="nav-link">Contact Us</a>.</p>
    </div>
    """

end

struct ProductsUsedComponent <: MultiDocumenter.DropdownComponent end

PRODUCTNAME_IMAGE_LINK = [
    (; product = "Dyad", logo = "https://help.juliahub.com/dyad/dev/dyad_logo_light.svg", link = "https://juliahub.com/products/dyad"),
    (; product = "Pumas", logo = "https://juliahub.com/ui/Pumas%20Logomark.svg", link = "https://pumas.ai/"),
    (; product = "Cedar EDA", logo = "https://juliahub.com/ui/cedar_eda.svg", link = "https://juliahub.com/products/cedar-eda"),
    (; product = "Neuroblox", logo = "https://juliahub.com/ui/Neuroblox-logo-400-300-dark.png", link = "https://www.neuroblox.org/"),
    (; product = "Planting Space", logo = "https://planting.space/img/logo_big.svg", link = "https://planting.space/"),
]

function MultiDocumenter.render(c::ProductsUsedComponent, doc, thispage, prettyurls)
    strings = [
        MultiDocumenter.@htl """
        <li>
            <a href=$(product.link) class="nav-link nav-item">
                $(product.product)
            </a>
        </li>
        """ for product in PRODUCTNAME_IMAGE_LINK
    ]

    return MultiDocumenter.@htl """
    <ul class="sciml-products">
        $strings
    </ul>
    """
end


struct Link <: MultiDocumenter.DropdownComponent
    text::String
    link::String
end

function MultiDocumenter.render(c::Link, doc, thispage, prettyurls)
    return MultiDocumenter.@htl """
    <a href=$(c.link) class="nav-link nav-item">$(c.text)</a>
    """
end
