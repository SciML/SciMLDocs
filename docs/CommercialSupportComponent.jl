struct JuliaHubCommercialSupportComponent <: MultiDocumenter.DropdownComponent
    link::String
end

function MultiDocumenter.render(c::JuliaHubCommercialSupportComponent, doc, thispage, prettyurls)

    return MultiDocumenter.@htl """
    <div style="color: #fff;">
        <a href="https://juliahub.com/products/dyad" class="nav-link nav-item" style="display: block; padding: 12px; border: 1px solid #5e6d6f; border-radius: 5px;">
            <img src="/assets/dyad-logo.svg" alt="Dyad" width="64" height="64" style="color-scheme: dark;"/>
            <p>Dyad by JuliaHub is the agentic modeling and simulation platform built on the SciML stack.</p>
            <ul style="list-style: disc; padding-left: 1.25rem; margin: 0;">
                <li>Component-based GUI for modeling</li>
                <li>Modeling language and compiler stack integrated with and transparent to AI agents</li>
            </ul>
        </a>
        <p style="margin-top: 12px;">JuliaHub offers commercial support for ModelingToolkit and the SciML ecosystem.
        <a href="$(c.link)" class="nav-link" style="display: inline; width: auto; padding: 0; text-decoration: underline;">Contact Us</a>.</p>
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
