struct JuliaHubCommercialSupportComponent <: MultiDocumenter.DropdownComponent
    link::String
end

function MultiDocumenter.render(c::JuliaHubCommercialSupportComponent, doc, thispage, prettyurls)

    return MultiDocumenter.@htl """
    <div style="color: #fff;">
        <a href="https://juliahub.com/products/dyad" class="nav-link nav-item">
            <img src="/assets/juliahub-logo.svg" alt="JuliaHub" style = "padding: 10px;"/>
        </a>
        <p>Dyad by JuliaHub is the agentic modeling and simulation platform built on the SciML stack.
        It helps you build, simulate, and validate physics-based models using natural language.</p>
        <a href="https://juliahub.com/products/dyad" class="nav-link nav-item">Explore Dyad</a>
        <p>JuliaHub offers commercial support for ModelingToolkit and the SciML ecosystem.</p>
        <a href="$(c.link)" class="nav-link nav-item">Contact JuliaHub for commercial support</a>
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
    <table>
        $strings
    </table>
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
